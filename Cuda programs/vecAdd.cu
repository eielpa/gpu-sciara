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
    // C[i+100000000] = A[i] + B[i]; per l'invalid memory access

}

void vecAdd(float* C, float* A, float* B, int n)
{
	float *d_C, *d_B, *d_A;
    int size = n* sizeof(float); //d sta per device, h per host
	int block_size = 1024, number_of_blocks = ceil((float)n/block_size); //32 è il numero minimo di thread da considerare per essere efficiente
	//block_size>1024 restituisce invalid argument
    cudaMalloc((void**)&d_A, size); //malloc come in C normalmente
    // cudaMalloc((void**)&d_A, size*9999999999999999999); //dà errore perchè il malloc è troppo grande
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);
    // d_A=nullptr; //rende d_A non allocato, quindi non può essere copiato 
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); //memcpy verso il device
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); //convenzione di C: il parametro target è sempre il primo (d_A/d_B sono all'inizio)
	vecAddKernel<<<number_of_blocks+1, block_size>>>(d_C, d_A, d_B, n); //<-- Async
    
    /*cudaError_t err = cudaDeviceSynchronize();
    if(err!=cudaSuccess){
        cout<<"CUDA error: "<<cudaGetErrorString(err)<<endl;
    }*/
    cudaDeviceSynchronize(); 
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToDevice); 
    //cambiando puntatore, es. devicetodevice, restituisce "invalid argument"
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        cout<<"CUDA error: "<<cudaGetErrorString(err)<<endl;
    }

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); //<-- Async
    
}

int main(){

    float A[]={0,1,2,3,4};
    float B[]={5,6,7,8,9};
    float C[5];
    int n=5;
    
    vecAdd(C,A,B,n);

    for(int i=0; i<n; i++){
        cout<<C[i]<<endl;
    }
    
    return 0;


}