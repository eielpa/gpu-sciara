#include <iostream>
#include <cmath>

using namespace std;

#define TILE_SIZE 8

__global__
void matrixInit(float* A, float* B, float* C, int sizeA, int sizeB, int sizeC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sizeA) {
        A[idx] = 0.5f / 1024.0f;
    }

    if (idx < sizeB) {
        B[idx] = 2.0f;
    }

    if (idx < sizeC) {
        C[idx] = 0.0f;
    }
}

__global__
void matrixMulBasic(const float* A, const float* B, float* C,
                    int rowsA, int colsA, int colsB) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

__global__ void matrixMulTiled(const float* A, const float* B, float* C,
                               int rowsA, int colsA, int colsB) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Numero di fasi (quante "tile" servono per coprire colsA)
    int numTiles = (colsA + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiledColA = t * TILE_SIZE + threadIdx.x;   // colonna di A
        int tiledRowB = t * TILE_SIZE + threadIdx.y;   // riga di B

        // Caricamento in shared memory, con controllo dei limiti
        if (row < rowsA && tiledColA < colsA)
            As[threadIdx.y][threadIdx.x] = A[row * colsA + tiledColA];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (tiledRowB < colsA && col < colsB)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRowB * colsB + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Calcolo parziale della tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Scrittura del risultato finale
    if (row < rowsA && col < colsB)
        C[row * colsB + col] = sum;
}

void checkCorrectness(const float* C, int rowsC, int colsC) {
    bool correct = true;
    float expected = 1.0f;

    for (int i = 0; i < rowsC * colsC; ++i) {
        if (fabs(C[i] - expected) > 1e-5) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, C[i]);
            correct = false;
            break;
        }
    }

    if (correct){
        printf("Matrix multiplication is correct!\n");
        for(int i=0; i<rowsC*colsC; i++){
            if(i%rowsC==0){
                printf("%f ", C[i]);
            }
        }
    }
}


int main() {

    int rowsA = 2048, colsA = 1024;
    int rowsB = colsA, colsB = 512;
    int rowsC = rowsA, colsC = colsB;

    int sizeA = rowsA * colsA;
    int sizeB = rowsB * colsB;
    int sizeC = rowsC * colsC;

    // Allocazione memoria managed
    float *A, *B, *C;
    cudaMallocManaged(&A, sizeA * sizeof(float));
    cudaMallocManaged(&B, sizeB * sizeof(float));
    cudaMallocManaged(&C, sizeC * sizeof(float));

    // --- Inizializzazione matrici GPU-side ---
    int blockSize1D = 256;
    int maxSize = max(sizeA, max(sizeB, sizeC));
    int gridSize1D = (maxSize + blockSize1D - 1) / blockSize1D;

    matrixInit<<<gridSize1D, blockSize1D>>>(A, B, C, sizeA, sizeB, sizeC);
    cudaDeviceSynchronize();

    // --- Verifica inizializzazione ---
    printf("A: "); for(int i = 0; i < 5; i++) printf("%f ", A[i]); printf("\n");
    printf("B: "); for(int i = 0; i < 5; i++) printf("%f ", B[i]); printf("\n");
    printf("C: "); for(int i = 0; i < 5; i++) printf("%f ", C[i]); printf("\n");

    // --- Configurazione 2D per moltiplicazione ---
    dim3 blockSize2D(TILE_SIZE, TILE_SIZE);
    dim3 gridSize2D((colsC + TILE_SIZE - 1) / TILE_SIZE,
                    (rowsC + TILE_SIZE - 1) / TILE_SIZE);

    // --- Basic matrix multiplication ---
    matrixMulBasic<<<gridSize2D, blockSize2D>>>(A, B, C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    printf("\n[Basic Kernel]\n");
    checkCorrectness(C, rowsC, colsC);

    // --- Reset matrice C ---
    cudaMemset(C, 0, sizeC * sizeof(float));

    // --- Tiled matrix multiplication ---
    matrixMulTiled<<<gridSize2D, blockSize2D>>>(A, B, C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    printf("\n[Tiled Kernel]\n");
    checkCorrectness(C, rowsC, colsC);

    // --- Libera memoria ---
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
