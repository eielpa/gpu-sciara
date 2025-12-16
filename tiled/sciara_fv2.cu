#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <algorithm>
#include <stdio.h>

// --- CONFIGURAZIONE TILING ---
#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif

// La Shared Memory deve ospitare il Tile + 1 cella di Halo su ogni lato
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

// Helper device per lettura sicura da Global con controllo bordi
__device__ double get_safe(double *M, int rows, int cols, int r, int c, double fallback) {
    if (r >= 0 && r < rows && c >= 0 && c < cols) return M[r * cols + c];
    return fallback;
}

// ----------------------------------------------------------------------------
// KERNEL TILED_wH (With Halo): computeOutflows
// ----------------------------------------------------------------------------
__global__ void computeOutflows_tiled_wH_kernel(
    int rows, int cols, int *Xi, int *Xj, double *Sz, double *Sh, double *ST, double *Mf,
    double Pc, double _a, double _b, double _c, double _d)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx; 
    int i = blockIdx.y * blockDim.y + ty; 

    // Shared Memory estesa (18x18)
    __shared__ double s_Sz[SHARED_H][SHARED_W];
    __shared__ double s_Sh[SHARED_H][SHARED_W];
    __shared__ double s_ST[SHARED_H][SHARED_W];

    // Coordinate in Shared (spostate di +1, +1 per lasciare spazio all'halo)
    int s_c = tx + 1;
    int s_r = ty + 1;

    // 1. CARICAMENTO CENTRO
    // Carico sempre, anche se fuori dai bordi globali (metto 0 o dummy) per evitare if nel calcolo dopo
    s_Sz[s_r][s_c] = get_safe(Sz, rows, cols, i, j, 0.0);
    s_Sh[s_r][s_c] = get_safe(Sh, rows, cols, i, j, 0.0);
    s_ST[s_r][s_c] = get_safe(ST, rows, cols, i, j, 0.0);

    // 2. CARICAMENTO HALO (Fatto dai thread di bordo)
    // Halo Superiore
    if (ty == 0) {
        s_Sz[0][s_c] = get_safe(Sz, rows, cols, i - 1, j, 0.0);
        s_Sh[0][s_c] = get_safe(Sh, rows, cols, i - 1, j, 0.0);
        // s_ST non serve per i vicini in questo kernel, ma per coerenza potremmo caricarlo
        // In computeOutflows serve l'altezza (z+h) dei vicini
    }
    // Halo Inferiore
    if (ty == TILE_H - 1) {
        s_Sz[SHARED_H - 1][s_c] = get_safe(Sz, rows, cols, i + 1, j, 0.0);
        s_Sh[SHARED_H - 1][s_c] = get_safe(Sh, rows, cols, i + 1, j, 0.0);
    }
    // Halo Sinistro
    if (tx == 0) {
        s_Sz[s_r][0] = get_safe(Sz, rows, cols, i, j - 1, 0.0);
        s_Sh[s_r][0] = get_safe(Sh, rows, cols, i, j - 1, 0.0);
    }
    // Halo Destro
    if (tx == TILE_W - 1) {
        s_Sz[s_r][SHARED_W - 1] = get_safe(Sz, rows, cols, i, j + 1, 0.0);
        s_Sh[s_r][SHARED_W - 1] = get_safe(Sh, rows, cols, i, j + 1, 0.0);
    }

    // Halo Angoli (4 thread specifici)
    if (tx == 0 && ty == 0) { // Top-Left
        s_Sz[0][0] = get_safe(Sz, rows, cols, i - 1, j - 1, 0.0);
        s_Sh[0][0] = get_safe(Sh, rows, cols, i - 1, j - 1, 0.0);
    }
    if (tx == TILE_W - 1 && ty == 0) { // Top-Right
        s_Sz[0][SHARED_W - 1] = get_safe(Sz, rows, cols, i - 1, j + 1, 0.0);
        s_Sh[0][SHARED_W - 1] = get_safe(Sh, rows, cols, i - 1, j + 1, 0.0);
    }
    if (tx == 0 && ty == TILE_H - 1) { // Bottom-Left
        s_Sz[SHARED_H - 1][0] = get_safe(Sz, rows, cols, i + 1, j - 1, 0.0);
        s_Sh[SHARED_H - 1][0] = get_safe(Sh, rows, cols, i + 1, j - 1, 0.0);
    }
    if (tx == TILE_W - 1 && ty == TILE_H - 1) { // Bottom-Right
        s_Sz[SHARED_H - 1][SHARED_W - 1] = get_safe(Sz, rows, cols, i + 1, j + 1, 0.0);
        s_Sh[SHARED_H - 1][SHARED_W - 1] = get_safe(Sh, rows, cols, i + 1, j + 1, 0.0);
    }

    __syncthreads(); // Barriera fondamentale

    // Check limiti e uscita rapida (leggendo da Shared)
    if (i >= rows || j >= cols) return;
    if (s_Sh[s_r][s_c] <= 0.0) return;

    // 3. CALCOLO (Tutto in Shared Memory)
    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS], h[MOORE_NEIGHBORS], H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS], w[MOORE_NEIGHBORS], Pr[MOORE_NEIGHBORS];
    double sz0, sz, T, avg, rr, hc;

    T = s_ST[s_r][s_c];
    rr = pow(10.0, _a + _b * T);
    hc = pow(10.0, _c + _d * T);
    sz0 = s_Sz[s_r][s_c];

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        // Accesso diretto in Shared usando gli offset Xi, Xj
        // Non serve nessun controllo di boundary qui, l'halo copre tutto!
        int n_r = s_r + Xj[k]; // Nota: Xj mappa sulle righe locali (y) o viceversa?
        int n_c = s_c + Xi[k]; // Xi mappa sulle colonne locali (x)
        // ATTENZIONE: Nel codice originale Xi è variazione riga o colonna?
        // Controlliamo: init() usa _Xi={0,-1..} e _Xj={0,0..}. Di solito i è riga, j è colonna.
        // Nello standard: ni = i + Xi[k]; nj = j + Xj[k];
        // Quindi Xi è variazione su i (righe), Xj su j (colonne).
        // Nel kernel: ty è riga, tx è colonna.
        // Quindi: n_ty = ty + Xi[k]; n_tx = tx + Xj[k];
        
        // Correggiamo gli indici shared:
        n_r = s_r + Xi[k]; // variazione riga
        n_c = s_c + Xj[k]; // variazione colonna

        sz = s_Sz[n_r][n_c];
        h[k] = s_Sh[n_r][n_c];

        w[k] = Pc; Pr[k] = rr;
        if (k < VON_NEUMANN_NEIGHBORS) z[k] = sz; else z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
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

    double h0 = s_Sh[s_r][s_c];
    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        double flow_val = 0.0;
        if (!eliminated[k] && h0 > hc * cos(theta[k])) flow_val = Pr[k] * (avg - H[k]);
        BUF_SET(Mf, rows, cols, k - 1, i, j, flow_val);
    }
}

// ----------------------------------------------------------------------------
// KERNEL TILED_wH: massBalance
// ----------------------------------------------------------------------------
__global__ void massBalance_tiled_wH_kernel(
    int rows, int cols, int *Xi, int *Xj, double *Sh, double *Sh_next, double *ST, double *ST_next, double *Mf)
{
    int tx = threadIdx.x; int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx; 
    int i = blockIdx.y * blockDim.y + ty; 

    // In MassBalance serve ST dei vicini
    __shared__ double s_ST[SHARED_H][SHARED_W];
    // Sh serve solo del centro? No, serve per i flussi... aspetta.
    // MassBalance: h_next = h + inFlows - outFlows.
    // InFlow viene da (ni, nj) verso ME. OutFlow viene da ME verso (ni, nj).
    // I flussi sono in Mf (Global).
    // L'unica variabile di stato del vicino che serve è ST (per calcolare la temperatura media ponderata).
    
    int s_c = tx + 1;
    int s_r = ty + 1;

    // Carico Centro
    s_ST[s_r][s_c] = get_safe(ST, rows, cols, i, j, 0.0);

    // Carico Halo per ST
    if (ty == 0) s_ST[0][s_c] = get_safe(ST, rows, cols, i - 1, j, 0.0);
    if (ty == TILE_H - 1) s_ST[SHARED_H - 1][s_c] = get_safe(ST, rows, cols, i + 1, j, 0.0);
    if (tx == 0) s_ST[s_r][0] = get_safe(ST, rows, cols, i, j - 1, 0.0);
    if (tx == TILE_W - 1) s_ST[s_r][SHARED_W - 1] = get_safe(ST, rows, cols, i, j + 1, 0.0);
    
    // Angoli
    if (tx == 0 && ty == 0) s_ST[0][0] = get_safe(ST, rows, cols, i - 1, j - 1, 0.0);
    if (tx == TILE_W - 1 && ty == 0) s_ST[0][SHARED_W - 1] = get_safe(ST, rows, cols, i - 1, j + 1, 0.0);
    if (tx == 0 && ty == TILE_H - 1) s_ST[SHARED_H - 1][0] = get_safe(ST, rows, cols, i + 1, j - 1, 0.0);
    if (tx == TILE_W - 1 && ty == TILE_H - 1) s_ST[SHARED_H - 1][SHARED_W - 1] = get_safe(ST, rows, cols, i + 1, j + 1, 0.0);

    __syncthreads();

    if (i >= rows || j >= cols) return;

    const int inflowsIndices[8] = {3, 2, 1, 0, 6, 7, 4, 5};
    double inFlow, outFlow, neigh_t;
    double initial_h = GET(Sh, cols, i, j); // Sh locale lo leggo da global (un solo accesso, va bene)
    double initial_t = s_ST[s_r][s_c];      // Da Shared
    
    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = i + Xi[n]; 
        int nj = j + Xj[n];
        
        int n_r = s_r + Xi[n];
        int n_c = s_c + Xj[n];

        // Lettura Temperatura vicino da Shared (Halo incluso)
        neigh_t = s_ST[n_r][n_c];

        // Mf è in Global
        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
            inFlow = BUF_GET(Mf, rows, cols, inflowsIndices[n - 1], ni, nj);
        } else inFlow = 0.0;

        outFlow = BUF_GET(Mf, rows, cols, n - 1, i, j);
        
        h_next += (inFlow - outFlow);
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0.0) {
        t_next /= h_next; SET(ST_next, cols, i, j, t_next); SET(Sh_next, cols, i, j, h_next);
    } else {
        SET(ST_next, cols, i, j, 0.0); SET(Sh_next, cols, i, j, 0.0);
    }
}

// ----------------------------------------------------------------------------
// KERNELS STANDARD (Invariati)
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

// ----------------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------------
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
  cudaMalloc(&d_total_emitted_lava, sizeof(double));
  cudaMemset(d_total_emitted_lava, 0, sizeof(double));

  double *d_total_current_lava = NULL;
  cudaMalloc(&d_total_current_lava, sizeof(double));
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

  printf("Simulation started (TILED_wH Version). Max Steps: %d\n", max_steps);
  cudaError_t err;

  // NOTA LE PARENTESI E L'OPERATORE &&
while ( (max_steps > 0 && sciara->simulation->step < max_steps) && 
        ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) || 
         (total_current_lava == -1 || total_current_lava > thickness_threshold)) )
  {
    sciara->simulation->elapsed_time += sciara->parameters->Pclock;
    sciara->simulation->step++;

    emitLava_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, sciara->simulation->d_vents, sciara->simulation->num_vents,
        sciara->simulation->elapsed_time, sciara->parameters->Pclock, sciara->simulation->emission_time, d_total_emitted_lava,
        sciara->parameters->Pac, sciara->parameters->PTvent, d_Sh, d_Sh, d_ST 
    );
    cudaDeviceSynchronize(); 
    err = cudaGetLastError(); if(err != cudaSuccess) { printf("ERR(emit): %s\n", cudaGetErrorString(err)); exit(1); }

    // TILED WITH HALO
    computeOutflows_tiled_wH_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, sciara->X->Xi, sciara->X->Xj, d_Sz, d_Sh, d_ST, sciara->substates->Mf,
        sciara->parameters->Pc, sciara->parameters->a, sciara->parameters->b, sciara->parameters->c, sciara->parameters->d
    );

    // TILED WITH HALO
    massBalance_tiled_wH_kernel<<<numBlocks, threadsPerBlock>>>(
        sciara->domain->rows, sciara->domain->cols, sciara->X->Xi, sciara->X->Xj, d_Sh, d_Sh_next, d_ST, d_ST_next, sciara->substates->Mf
    );
    std::swap(d_Sh, d_Sh_next); std::swap(d_ST, d_ST_next);

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

  cudaFree(d_total_emitted_lava);
  cudaFree(d_total_current_lava);
  finalize(sciara);

  return 0;
}