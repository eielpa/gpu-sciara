#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Kernel CUDA
__global__ void vecAdd(float *A, float *B, float *C, int N, int grain) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcola l'intervallo di elementi che questo thread deve gestire
    int start = global_thread_id * grain;
    int end = start + grain;

    // Somma gli elementi assegnati a questo thread
    for (int i = start; i < end && i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N;
    cout<<"Insert vec size: ";
    cin>>N;
    int grain;
    cout<<"Insert number of elements for thread: ";
    cin>>grain;

    size_t size = N * sizeof(float);

    // Alloca memoria host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Inizializza i dati
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Alloca memoria device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copia dati su GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Calcola numero di thread totali necessari
    int threadsPerBlock = 256;
    int numThreadsTotal = (N + grain - 1) / grain;  // ogni thread fa "grain" operazioni
    int numBlocks = (numThreadsTotal + threadsPerBlock - 1) / threadsPerBlock;

    // Lancia kernel
    vecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, grain);

    // Copia risultati su host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verifica alcuni risultati
    for(int i=0; i<N;i++){
        printf("%f", h_C[i]);
    }

    // Libera memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
