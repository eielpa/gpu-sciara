#include "Sciara.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath> 

#define GET_CPU(M, cols, r, c) M[(r)*(cols) + (c)]
#define SET_CPU(M, cols, r, c, val) M[(r)*(cols) + (c)] = val

void allocateSubstates(Sciara *sciara)
{
    size_t size_double = sciara->domain->rows * sciara->domain->cols * sizeof(double);
    size_t size_bool   = sciara->domain->rows * sciara->domain->cols * sizeof(bool);
    
    cudaMallocManaged(&(sciara->substates->Sz), size_double);
    cudaMallocManaged(&(sciara->substates->Sz_next), size_double);
    cudaMallocManaged(&(sciara->substates->Sh), size_double);
    cudaMallocManaged(&(sciara->substates->Sh_next), size_double);
    cudaMallocManaged(&(sciara->substates->ST), size_double);
    cudaMallocManaged(&(sciara->substates->ST_next), size_double);
    
    size_t size_flows = sciara->domain->rows * sciara->domain->cols * NUMBER_OF_OUTFLOWS * sizeof(double);
    cudaMallocManaged(&(sciara->substates->Mf), size_flows);

    cudaMallocManaged(&(sciara->substates->Mb), size_bool);
    cudaMallocManaged(&(sciara->substates->Mhs), size_double);
    
    cudaMemset(sciara->substates->Sz, 0, size_double);
    cudaMemset(sciara->substates->Sh, 0, size_double);
    cudaMemset(sciara->substates->ST, 0, size_double);
    cudaMemset(sciara->substates->Mf, 0, size_flows);
    cudaMemset(sciara->substates->Mb, 0, size_bool);
    cudaMemset(sciara->substates->Mhs, 0, size_double);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CRITICAL: CUDA Error in allocateSubstates: %s\n", cudaGetErrorString(err));
        exit(1);
    }
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

// ----------------------------------------------------------------------------
// MAKE BORDER (Versione sicura)
// ----------------------------------------------------------------------------
void makeBorder(Sciara *sciara) 
{
    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;
    double* Sz = sciara->substates->Sz;
    bool* Mb = sciara->substates->Mb;

    for(int i=0; i<rows*cols; i++) Mb[i] = false;

    // Bordi esterni
    for (int j = 0; j < cols; j++) {
        if (GET_CPU(Sz, cols, 0, j) >= 0) SET_CPU(Mb, cols, 0, j, true);
        if (GET_CPU(Sz, cols, rows-1, j) >= 0) SET_CPU(Mb, cols, rows-1, j, true);
    }

    for (int i = 0; i < rows; i++) {
        if (GET_CPU(Sz, cols, i, 0) >= 0) SET_CPU(Mb, cols, i, 0, true);
        if (GET_CPU(Sz, cols, i, cols-1) >= 0) SET_CPU(Mb, cols, i, cols-1, true);
    }
  
    // Bordi interni
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (GET_CPU(Sz, cols, i, j) >= 0) {
                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                    int ni = i + sciara->X->Xi[k];
                    int nj = j + sciara->X->Xj[k];
                    if (GET_CPU(Sz, cols, ni, nj) < 0) {
                        SET_CPU(Mb, cols, i, j, true);
                        break;
                    }
                }
            }
        }
    }
}

void simulationInitialize(Sciara* sciara)
{ 
  unsigned int maximum_number_of_emissions = 0;

  sciara->simulation->step = 0;
  sciara->simulation->elapsed_time = 0;

  for (unsigned int i = 0; i < sciara->simulation->emission_rate.size(); i++)
    if (maximum_number_of_emissions < sciara->simulation->emission_rate[i].size())
      maximum_number_of_emissions = sciara->simulation->emission_rate[i].size();
  
  sciara->simulation->effusion_duration = sciara->simulation->emission_time * maximum_number_of_emissions;
  sciara->simulation->total_emitted_lava = 0;

  makeBorder(sciara);

  evaluatePowerLawParams(sciara->parameters->PTvent, sciara->parameters->PTsol, sciara->parameters->Pr_Tsol, sciara->parameters->Pr_Tvent, sciara->parameters->a, sciara->parameters->b);
  evaluatePowerLawParams(sciara->parameters->PTvent, sciara->parameters->PTsol, sciara->parameters->Phc_Tsol, sciara->parameters->Phc_Tvent, sciara->parameters->c, sciara->parameters->d);

  // Preparazione Vents su GPU
  sciara->simulation->num_vents = sciara->simulation->vent.size();

  if (sciara->simulation->num_vents > 0) {
      cudaMallocManaged(&(sciara->simulation->d_vents), sciara->simulation->num_vents * sizeof(CudaVent));

      for(int k = 0; k < sciara->simulation->num_vents; k++) 
      {
          sciara->simulation->d_vents[k].x = sciara->simulation->vent[k].x();
          sciara->simulation->d_vents[k].y = sciara->simulation->vent[k].y();
          
          int num_rates = sciara->simulation->vent[k].size();
          if (num_rates > MAX_RATES) num_rates = MAX_RATES;
          
          sciara->simulation->d_vents[k].num_rates = num_rates;

          for(int r = 0; r < num_rates; r++) {
              sciara->simulation->d_vents[k].rates[r] = sciara->simulation->vent[k][r];
          }
      }
  } else {
      sciara->simulation->d_vents = NULL;
  }
}

int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; 
int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; 

void init(Sciara*& sciara)
{
  sciara = new Sciara;
  sciara->domain = new Domain;
  sciara->X = new NeighsRelativeCoords;
  
  cudaMallocManaged(&(sciara->X->Xi), MOORE_NEIGHBORS * sizeof(int));
  cudaMallocManaged(&(sciara->X->Xj), MOORE_NEIGHBORS * sizeof(int));

  for (int n=0; n<MOORE_NEIGHBORS; n++)
  {
    sciara->X->Xi[n] = _Xi[n];
    sciara->X->Xj[n] = _Xj[n];
  }

  sciara->substates = new Substates; 
  sciara->parameters = new Parameters;
  sciara->simulation = new Simulation;
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