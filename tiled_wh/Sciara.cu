#include "Sciara.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath> 

// Kernel per calcolare i bordi direttamente su GPU
__global__ void makeBorder_kernel(int rows, int cols, double* Sz, bool* Mb, int* Xi, int* Xj) 
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i >= rows || j >= cols) return;

    // Reset iniziale
    Mb[i * cols + j] = false;
    __syncthreads(); 

    // 1. Bordi esterni
    if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        if (Sz[i * cols + j] >= 0) Mb[i * cols + j] = true;
    }

    // 2. Bordi interni (buco nel DEM)
    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        if (Sz[i * cols + j] >= 0) {
            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                int ni = i + Xi[k];
                int nj = j + Xj[k];
                if (Sz[ni * cols + nj] < 0) {
                    Mb[i * cols + j] = true;
                    break;
                }
            }
        }
    }
}

void allocateSubstates(Sciara *sciara)
{
    printf("DEBUG: [allocateSubstates] Allocating on GPU...\n"); fflush(stdout);
    size_t size_double = sciara->domain->rows * sciara->domain->cols * sizeof(double);
    size_t size_bool   = sciara->domain->rows * sciara->domain->cols * sizeof(bool);
    
    cudaError_t err;
    err = cudaMalloc(&(sciara->substates->Sz), size_double);
    if(err!=cudaSuccess) printf("Alloc Error Sz\n");

    cudaMalloc(&(sciara->substates->Sz_next), size_double);
    cudaMalloc(&(sciara->substates->Sh), size_double);
    cudaMalloc(&(sciara->substates->Sh_next), size_double);
    cudaMalloc(&(sciara->substates->ST), size_double);
    cudaMalloc(&(sciara->substates->ST_next), size_double);
    
    size_t size_flows = sciara->domain->rows * sciara->domain->cols * NUMBER_OF_OUTFLOWS * sizeof(double);
    cudaMalloc(&(sciara->substates->Mf), size_flows);

    cudaMalloc(&(sciara->substates->Mb), size_bool);
    cudaMalloc(&(sciara->substates->Mhs), size_double);
    
    // Reset memoria GPU
    cudaMemset(sciara->substates->Sz, 0, size_double);
    cudaMemset(sciara->substates->Sh, 0, size_double);
    cudaMemset(sciara->substates->ST, 0, size_double);
    cudaMemset(sciara->substates->Mf, 0, size_flows);
    cudaMemset(sciara->substates->Mb, 0, size_bool);
    cudaMemset(sciara->substates->Mhs, 0, size_double);
    
    printf("DEBUG: [allocateSubstates] Done.\n"); fflush(stdout);
}

void deallocateSubstates(Sciara *sciara)
{
    if(sciara->substates->Sz)       cudaFree(sciara->substates->Sz);
    if(sciara->substates->Sz_next)  cudaFree(sciara->substates->Sz_next);
    if(sciara->substates->Sh)       cudaFree(sciara->substates->Sh);
    if(sciara->substates->Sh_next)  cudaFree(sciara->substates->Sh_next);
    if(sciara->substates->ST)       cudaFree(sciara->substates->ST);
    if(sciara->substates->ST_next)  cudaFree(sciara->substates->ST_next);
    if(sciara->substates->Mf)       cudaFree(sciara->substates->Mf);
    if(sciara->substates->Mb)       cudaFree(sciara->substates->Mb);
    if(sciara->substates->Mhs)      cudaFree(sciara->substates->Mhs);
}

void evaluatePowerLawParams(double PTvent, double PTsol, double value_sol, double value_vent, double &k1, double &k2)
{
	k2 = ( log10(value_vent) - log10(value_sol) ) / (PTvent - PTsol) ;
	k1 = log10(value_sol) - k2*(PTsol);
}

void simulationInitialize(Sciara* sciara)
{ 
  printf("DEBUG: [simulationInitialize] Start.\n"); fflush(stdout);
  unsigned int maximum_number_of_emissions = 0;

  sciara->simulation->step = 0;
  sciara->simulation->elapsed_time = 0;

  for (unsigned int i = 0; i < sciara->simulation->emission_rate.size(); i++)
    if (maximum_number_of_emissions < sciara->simulation->emission_rate[i].size())
      maximum_number_of_emissions = sciara->simulation->emission_rate[i].size();
  
  sciara->simulation->effusion_duration = sciara->simulation->emission_time * maximum_number_of_emissions;
  sciara->simulation->total_emitted_lava = 0;

  printf("DEBUG: [simulationInitialize] Launching makeBorder_kernel...\n"); fflush(stdout);
  dim3 threads(16, 16);
  dim3 blocks((sciara->domain->cols + 15) / 16, (sciara->domain->rows + 15) / 16);
  makeBorder_kernel<<<blocks, threads>>>(
      sciara->domain->rows, 
      sciara->domain->cols, 
      sciara->substates->Sz, 
      sciara->substates->Mb, 
      sciara->X->Xi, 
      sciara->X->Xj
  );
  cudaDeviceSynchronize();

  evaluatePowerLawParams(sciara->parameters->PTvent, sciara->parameters->PTsol, sciara->parameters->Pr_Tsol, sciara->parameters->Pr_Tvent, sciara->parameters->a, sciara->parameters->b);
  evaluatePowerLawParams(sciara->parameters->PTvent, sciara->parameters->PTsol, sciara->parameters->Phc_Tsol, sciara->parameters->Phc_Tvent, sciara->parameters->c, sciara->parameters->d);

  printf("DEBUG: [simulationInitialize] Preparing Vents GPU...\n"); fflush(stdout);
  sciara->simulation->num_vents = sciara->simulation->vent.size();

  if (sciara->simulation->num_vents > 0) {
      cudaMalloc(&(sciara->simulation->d_vents), sciara->simulation->num_vents * sizeof(CudaVent));
      
      CudaVent *h_vents = new CudaVent[sciara->simulation->num_vents];

      for(int k = 0; k < sciara->simulation->num_vents; k++) 
      {
          h_vents[k].x = sciara->simulation->vent[k].x();
          h_vents[k].y = sciara->simulation->vent[k].y();
          
          int num_rates = sciara->simulation->vent[k].size();
          if (num_rates > MAX_RATES) num_rates = MAX_RATES;
          
          h_vents[k].num_rates = num_rates;

          for(int r = 0; r < num_rates; r++) {
              h_vents[k].rates[r] = sciara->simulation->vent[k][r];
          }
      }
      cudaMemcpy(sciara->simulation->d_vents, h_vents, sciara->simulation->num_vents * sizeof(CudaVent), cudaMemcpyHostToDevice);
      delete[] h_vents;

  } else {
      sciara->simulation->d_vents = NULL;
  }
  printf("DEBUG: [simulationInitialize] Done.\n"); fflush(stdout);
}

int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; 
int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; 

void init(Sciara*& sciara)
{
  printf("DEBUG: [init] Start.\n"); fflush(stdout);
  sciara = new Sciara;
  sciara->domain = new Domain;
  sciara->X = new NeighsRelativeCoords;
  
  cudaMalloc(&(sciara->X->Xi), MOORE_NEIGHBORS * sizeof(int));
  cudaMalloc(&(sciara->X->Xj), MOORE_NEIGHBORS * sizeof(int));

  cudaMemcpy(sciara->X->Xi, _Xi, MOORE_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(sciara->X->Xj, _Xj, MOORE_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);

  sciara->substates = new Substates; 
  sciara->parameters = new Parameters;
  sciara->simulation = new Simulation;
  printf("DEBUG: [init] Done.\n"); fflush(stdout);
}

void finalize(Sciara*& sciara)
{
  deallocateSubstates(sciara);
  delete sciara->domain;
  cudaFree(sciara->X->Xi);
  cudaFree(sciara->X->Xj);
  delete sciara->X;
  delete sciara->substates;
  delete sciara->parameters;
  delete sciara->simulation;
  delete sciara;
  sciara = NULL;
}