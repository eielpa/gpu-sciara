#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <algorithm>
#include <stdio.h>

// --- CONFIGURAZIONE TILING ---
#ifndef TILE_W
#define TILE_W 32
#endif
#ifndef TILE_H
#define TILE_H 16
#endif

// Shared Memory con Halo (Tile + 2 bordi) per lettura veloce input
#define SHARED_W (TILE_W + 2)
#define SHARED_H (TILE_H + 2)

#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

// Helper per Atomic Add su Double (Necessario per Maxwell/GTX 980)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Helper lettura sicura
__device__ double get_safe(double *M, int rows, int cols, int r, int c, double fallback) {
    if (r >= 0 && r < rows && c >= 0 && c < cols) return M[r * cols + c];
    return fallback;
}

// Helper calcolo flussi locale (senza scrivere in memoria, ritorna solo valori)
__device__ void calculate_flows_cfamo(
    double h0, double z0, double T0, int s_r, int s_c, 
    int *Xi, int *Xj, double (*s_Sz)[SHARED_W], double (*s_h)[SHARED_W], 
    double Pc, double _a, double _b, double _c, double _d,
    double *out_flows) 
{
    for(int k=0; k<8; k++) out_flows[k] = 0.0;
    if (h0 <= 0.0) return;

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS], h[MOORE_NEIGHBORS], H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS], w[MOORE_NEIGHBORS], Pr[MOORE_NEIGHBORS];
    double avg, rr, hc;

    rr = pow(10.0, _a + _b * T0);
    hc = pow(10.0, _c + _d * T0);

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int nr = s_r + Xi[k];
        int nc = s_c + Xj[k];
        double sz = s_Sz[nr][nc];
        h[k] = s_h[nr][nc];
        
        w[k] = Pc; Pr[k] = rr;
        if (k < VON_NEUMANN_NEIGHBORS) z[k] = sz; else z[k] = z0 - (z0 - sz) / sqrt(2.0);
    }

    H[0] = z[0]; theta[0] = 0.0; eliminated[0] = false;
    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        if (z[0] + h[0] > z[k] + h[k]) {
            H[k] = z[k] + h[k]; theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]); eliminated[k] = false;
        } else eliminated[k] = true;
    }

    bool loop;
    do {
        loop = false; avg = h[0]; int counter = 0;
        for (int k = 0; k < MOORE_NEIGHBORS; k++) if (!eliminated[k]) { avg += H[k]; counter++; }
        if (counter != 0) avg = avg / double(counter);
        for (int k = 0; k < MOORE_NEIGHBORS; k++) if (!eliminated[k] && avg <= H[k]) { eliminated[k] = true; loop = true; }
    } while (loop);

    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        if (!eliminated[k] && h0 > hc * cos(theta[k])) {
            out_flows[k-1] = Pr[k] * (avg - H[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// KERNEL 1: Inizializzazione Accumulatori (Energia e Massa)
// ----------------------------------------------------------------------------
__global__ void CfAMo_init_kernel(int rows, int cols, double *Sh, double *ST, double *Sh_next, double *ST_next) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= rows || j >= cols) return;

    double h = GET(Sh, cols, i, j);
    double t = GET(ST, cols, i, j);
    
    // Inizializziamo il prossimo passo con lo stato attuale
    SET(Sh_next, cols, i, j, h);
    // ST_next viene usato temporaneamente per accumulare ENERGIA (h * t)
    SET(ST_next, cols, i, j, h * t);
}

// ----------------------------------------------------------------------------
// KERNEL 2: Calcolo e Aggiornamento Atomico
// ----------------------------------------------------------------------------
__global__ void CfAMo_update_kernel(
    int rows, int cols, int *Xi, int *Xj, double *Sz, double *Sh, double *ST, 
    double *Sh_next, double *ST_next, // Destinazioni (Accumulatori)
    double Pc, double _a, double _b, double _c, double _d)
{
    int tx = threadIdx.x; int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx; 
    int i = blockIdx.y * blockDim.y + ty; 

    // Shared Memory solo per lettura veloce INPUT (Tile+Halo)
    __shared__ double s_Sz[SHARED_H][SHARED_W];
    __shared__ double s_h[SHARED_H][SHARED_W];
    
    int s_c = tx + 1; int s_r = ty + 1;

    // Caricamento Input (Tiled_wH style)
    s_Sz[s_r][s_c] = get_safe(Sz, rows, cols, i, j, 0.0);
    s_h[s_r][s_c] = get_safe(Sh, rows, cols, i, j, 0.0);

    if (ty == 0) { s_Sz[0][s_c] = get_safe(Sz, rows, cols, i-1, j, 0.0); s_h[0][s_c] = get_safe(Sh, rows, cols, i-1, j, 0.0); }
    if (ty == TILE_H-1) { s_Sz[SHARED_H-1][s_c] = get_safe(Sz, rows, cols, i+1, j, 0.0); s_h[SHARED_H-1][s_c] = get_safe(Sh, rows, cols, i+1, j, 0.0); }
    if (tx == 0) { s_Sz[s_r][0] = get_safe(Sz, rows, cols, i, j-1, 0.0); s_h[s_r][0] = get_safe(Sh, rows, cols, i, j-1, 0.0); }
    if (tx == TILE_W-1) { s_Sz[s_r][SHARED_W-1] = get_safe(Sz, rows, cols, i, j+1, 0.0); s_h[s_r][SHARED_W-1] = get_safe(Sh, rows, cols, i, j+1, 0.0); }
    
    // Angoli
    if (tx==0 && ty==0) { s_Sz[0][0] = get_safe(Sz, rows, cols, i-1, j-1, 0.0); s_h[0][0] = get_safe(Sh, rows, cols, i-1, j-1, 0.0); }
    if (tx==TILE_W-1 && ty==0) { s_Sz[0][SHARED_W-1] = get_safe(Sz, rows, cols, i-1, j+1, 0.0); s_h[0][SHARED_W-1] = get_safe(Sh, rows, cols, i-1, j+1, 0.0); }
    if (tx==0 && ty==TILE_H-1) { s_Sz[SHARED_H-1][0] = get_safe(Sz, rows, cols, i+1, j-1, 0.0); s_h[SHARED_H-1][0] = get_safe(Sh, rows, cols, i+1, j-1, 0.0); }
    if (tx==TILE_W-1 && ty==TILE_H-1) { s_Sz[SHARED_H-1][SHARED_W-1] = get_safe(Sz, rows, cols, i+1, j+1, 0.0); s_h[SHARED_H-1][SHARED_W-1] = get_safe(Sh, rows, cols, i+1, j+1, 0.0); }

    __syncthreads();

    // Solo i thread nel dominio attivo calcolano e scrivono
    if (i >= rows || j >= cols) return;

    double my_T = GET(ST, cols, i, j); // Temp non serve in shared per il calcolo geometrico
    double my_flows[8];

    // Calcola flussi uscenti
    calculate_flows_cfamo(s_h[s_r][s_c], s_Sz[s_r][s_c], my_T, s_r, s_c, Xi, Xj, s_Sz, s_h, Pc, _a, _b, _c, _d, my_flows);

    // Aggiornamento Atomico in GLOBAL MEMORY (Cruciale per correttezza cross-block)
    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        double f = my_flows[k-1];
        if (f > 0.0) {
            int ni = i + Xi[k];
            int nj = j + Xj[k];
            
            // 1. Aggiungo a vicino (Massa e Energia)
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                // atomicAdd su Global Memory è l'unico modo sicuro senza kernel multipli
                atomicAdd(&GET(Sh_next, cols, ni, nj), f);
                atomicAdd(&GET(ST_next, cols, ni, nj), f * my_T);
            }

            // 2. Sottraggo a me stesso
            atomicAdd(&GET(Sh_next, cols, i, j), -f);
            atomicAdd(&GET(ST_next, cols, i, j), -f * my_T);
        }
    }
}

// ----------------------------------------------------------------------------
// KERNEL 3: Finalizzazione (Energia -> Temperatura)
// ----------------------------------------------------------------------------
__global__ void CfAMo_finalize_kernel(int rows, int cols, double *Sh_next, double *ST_next) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= rows || j >= cols) return;

    double h = GET(Sh_next, cols, i, j);
    double energy = GET(ST_next, cols, i, j);

    if (h > 0.000001) {
        SET(ST_next, cols, i, j, energy / h);
    } else {
        SET(ST_next, cols, i, j, 0.0);
        SET(Sh_next, cols, i, j, 0.0);
    }
}

// ----------------------------------------------------------------------------
// KERNELS STANDARD (Accessori)
// ----------------------------------------------------------------------------
__global__ void emitLava_kernel(
    int rows, int cols, CudaVent *vents, int num_vents, double elapsed_time, double Pclock, double emission_time,
    double *total_emitted_lava, double Pac, double PTvent, double *Sh, double *Sh_next, double *ST)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= rows || j >= cols) return;
    if (num_vents == 0 || vents == NULL) return;

    for (int k = 0; k < num_vents; k++) {
        if (i == vents[k].y && j == vents[k].x){
            double flow_thickness = 0.0;
            unsigned int idx = (unsigned int)(elapsed_time / emission_time);
            if (idx < vents[k].num_rates) flow_thickness = vents[k].rates[idx] / Pac * Pclock;
            if (flow_thickness > 0.0) {
                double current_h = GET(Sh, cols, i, j);
                SET(Sh, cols, i, j, current_h + flow_thickness); 
                SET(ST, cols, i, j, PTvent); 
                atomicAdd(total_emitted_lava, flow_thickness);
            }
        }
    }
}

__global__ void computeNewTemperatureAndSolidification_kernel(
    int rows, int cols, double Pepsilon, double Psigma, double Pclock, double Pcool, double Prho, double Pcv, double Pac, double PTsol,
    double *Sz, double *Sz_next, double *Sh, double *Sh_next, double *ST, double *ST_next, double *Mhs, bool *Mb)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= rows || j >= cols) return;

    double z = GET(Sz, cols, i, j), h = GET(Sh, cols, i, j), T = GET(ST, cols, i, j); bool is_boundary = GET(Mb, cols, i, j);
    SET(Sh_next, cols, i, j, h); SET(ST_next, cols, i, j, T); SET(Sz_next, cols, i, j, z);

    if (h > 0.000001 && !is_boundary) {
        double aus = 1.0 + (3.0 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
        double nT = T / pow(aus, 1.0 / 3.0);
        if (nT > PTsol) { SET(ST_next, cols, i, j, nT); }
        else { SET(Sz_next, cols, i, j, z + h); SET(Sh_next, cols, i, j, 0.0); SET(ST_next, cols, i, j, PTsol);
               double current_Mhs = GET(Mhs, cols, i, j); SET(Mhs, cols, i, j, current_Mhs + h); }
    }
}

__global__ void boundaryConditions_kernel(int rows, int cols, bool *Mb, double *Sh, double *Sh_next, double *ST, double *ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= rows || j >= cols) return;
    double h = GET(Sh, cols, i, j), t = GET(ST, cols, i, j);
    if (GET(Mb, cols, i, j)) { SET(Sh_next, cols, i, j, 0.0); SET(ST_next, cols, i, j, 0.0); } 
    else { SET(Sh_next, cols, i, j, h); SET(ST_next, cols, i, j, t); }
}

__global__ void globalReduction_kernel(int rows, int cols, double *Sh, double *output_sum)
{
    __shared__ double block_sum;
    if (threadIdx.x == 0 && threadIdx.y == 0) block_sum = 0.0;
    __syncthreads();
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < rows && j < cols) {
        double val = GET(Sh, cols, i, j);
        if (val > 0.0) atomicAdd(&block_sum, val);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(output_sum, block_sum);
}

int main(int argc, char **argv)
{
  cudaSetDevice(0);
  Sciara *sciara = NULL;
  init(sciara); 

  if (argc <= MAX_STEPS_ID) { printf("ERROR: Not enough arguments.\n"); exit(1); }
  if (loadConfiguration(argv[INPUT_PATH_ID], sciara) != 1) { printf("ERROR: loadConfiguration failed.\n"); exit(1); }
  
  double total_current_lava = -1;
  simulationInitialize(sciara);

  dim3 threadsPerBlock(TILE_W, TILE_H);
  dim3 numBlocks((sciara->domain->cols + TILE_W - 1) / TILE_W, (sciara->domain->rows + TILE_H - 1) / TILE_H);

  double *d_total_emitted_lava = NULL;
  cudaMallocManaged(&d_total_emitted_lava, sizeof(double));
  cudaMemset(d_total_emitted_lava, 0, sizeof(double));

  double *d_total_current_lava = NULL;
  cudaMallocManaged(&d_total_current_lava, sizeof(double));
  cudaMemset(d_total_current_lava, 0, sizeof(double));

  util::Timer cl_timer;
  int max_steps = atoi(argv[MAX_STEPS_ID]);
  int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
  double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

  double *d_Sh = sciara->substates->Sh;
  double *d_Sh_next = sciara->substates->Sh_next;
  double *d_ST = sciara->substates->ST;
  double *d_ST_next = sciara->substates->ST_next;
  double *d_Sz = sciara->substates->Sz;
  double *d_Sz_next = sciara->substates->Sz_next;

  printf("Simulation started (CfAMo Atomic Version). Max Steps: %d\n", max_steps);
  
  while ((max_steps > 0 && sciara->simulation->step < max_steps) || 
         (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) || 
         (total_current_lava == -1 || total_current_lava > thickness_threshold))
  {
    sciara->simulation->elapsed_time += sciara->parameters->Pclock;
    sciara->simulation->step++;

    emitLava_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, sciara->simulation->d_vents, sciara->simulation->num_vents,
        sciara->simulation->elapsed_time, sciara->parameters->Pclock, sciara->simulation->emission_time, d_total_emitted_lava,
        sciara->parameters->Pac, sciara->parameters->PTvent, d_Sh, d_Sh, d_ST 
    );
    cudaDeviceSynchronize(); 
    
    sciara->simulation->total_emitted_lava = *d_total_emitted_lava;

    // --- CfAMo SEQUENCE ---
    
    // 1. Init Accumulatori (Sh_next = Sh, ST_next = Energy)
    CfAMo_init_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, d_Sh, d_ST, d_Sh_next, d_ST_next
    );
    cudaDeviceSynchronize();

    // 2. Calcolo e Update Atomico
    CfAMo_update_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, sciara->X->Xi, sciara->X->Xj, 
        d_Sz, d_Sh, d_ST, d_Sh_next, d_ST_next,
        sciara->parameters->Pc, sciara->parameters->a, sciara->parameters->b, sciara->parameters->c, sciara->parameters->d
    );
    cudaDeviceSynchronize();

    // 3. Finalize (Energy -> Temperature)
    CfAMo_finalize_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, d_Sh_next, d_ST_next
    );
    cudaDeviceSynchronize();

    // Swap pointers
    std::swap(d_Sh, d_Sh_next);
    std::swap(d_ST, d_ST_next);

    // --- End CfAMo ---

    computeNewTemperatureAndSolidification_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, sciara->parameters->Pepsilon, sciara->parameters->Psigma,
        sciara->parameters->Pclock, sciara->parameters->Pcool, sciara->parameters->Prho, sciara->parameters->Pcv,
        sciara->parameters->Pac, sciara->parameters->PTsol, d_Sz, d_Sz_next, d_Sh, d_Sh_next, d_ST, d_ST_next,
        sciara->substates->Mhs, sciara->substates->Mb
    );
    std::swap(d_Sz, d_Sz_next); std::swap(d_Sh, d_Sh_next); std::swap(d_ST, d_ST_next);

    boundaryConditions_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, sciara->substates->Mb, d_Sh, d_Sh_next, d_ST, d_ST_next
    );
    std::swap(d_Sh, d_Sh_next); std::swap(d_ST, d_ST_next);

    if (sciara->simulation->step % reduceInterval == 0) {
      cudaMemset(d_total_current_lava, 0, sizeof(double));
      globalReduction_kernel<<<numBlocks, threadsPerBlock>>>(sciara->domain->rows, sciara->domain->cols, d_Sh, d_total_current_lava);
      cudaDeviceSynchronize();
      cudaMemcpy(&total_current_lava, d_total_current_lava, sizeof(double), cudaMemcpyDeviceToHost);
      printf("Step: %d | Lava: %.2f | Time: %.2f s\n", sciara->simulation->step, total_current_lava, sciara->simulation->elapsed_time);
      fflush(stdout);
    }
  }
  
  sciara->substates->Sh = d_Sh;
  sciara->substates->ST = d_ST;
  sciara->substates->Sz = d_Sz;

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Final Step %d. Total Time: %lf s\n", sciara->simulation->step, cl_time);
  printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
  saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

  cudaFree(d_total_emitted_lava); cudaFree(d_total_current_lava); finalize(sciara);
  return 0;
}