#include <iostream>
using namespace std;

__global__
void MatrixMulKernel(float* P, float* M, float* N, int Width){
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((Row < Width) && (Col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row*Width+k] * N[k*Width+Col];
        }
        P[Row*Width+Col] = Pvalue;
    }
}

int main(){

    int width=64;
    int size=width*width*sizeof(float);

    float *P, *M, *N;

    cudaMallocManaged(&P, size);
    cudaMallocManaged(&M, size);
    cudaMallocManaged(&N, size);

    for(int i=0; i<width*width; i++){
        M[i]=i;
        N[i]=i+1;
    }

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
        (width + dimBlock.y - 1) / dimBlock.y);


    MatrixMulKernel<<<dimGrid, dimBlock>>>(P, M, N, width);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    for(int i=0; i<16; i++){
        printf("%f \n", P[i]);
    }

    cudaFree(P);
    cudaFree(M);
    cudaFree(N);

}
