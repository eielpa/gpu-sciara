#include <iostream>
using namespace std;

// The serial loop is replaced by the pool of threads
// Each thread performs one pair-wise addition
__global__ void vecAddKernel(float* C, float* A, float* B, int n)
{
//i is a private variable
int i = blockDim.x*blockIdx.x + threadIdx.x;
if (i < n) // to be sure to "intercept" data
C[i] = A[i] + B[i];
}

void vecAdd(float* C, float* A, float* B, int n)
{
    int size = n * sizeof(float);
    int block_size = 32;
    int number_of_blocks = ceil((float)n / block_size);

    // Allocazione unificata
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Inizializzazione dei dati (CPU)
    for (int i = 0; i < n; i++) {
        A[i] = i; // valori iniziali
        B[i] = i+n;
    }

    // Esecuzione kernel
    vecAddKernel<<<number_of_blocks, block_size>>>(C, A, B, n);

    // Attendere che la GPU termini
    cudaDeviceSynchronize();

    // (A questo punto C[i] contiene i risultati)
    // Eventuale stampa o controllo su CPU
    for (int i = 0; i < n; i++) {
        cout<<C[i]<<endl;
    }

    // Liberare la memoria
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main(){

    float *A, *B, *C;
    int n=5;

    vecAdd(C,A,B,n);
    
    return 0;

}