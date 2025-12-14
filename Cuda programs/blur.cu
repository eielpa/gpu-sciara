#include <iostream>
#include <vector>
#include "lodepng.h"
#include <cuda_runtime.h>

using namespace std;

#define BLUR_SIZE 5

__global__ void blurKernel(unsigned char* out, unsigned char* in, int w, int h)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixR = 0; int pixG = 0; int pixB = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++)
        for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++)
        {
            int curRow = Row + blurRow;
            int curCol = Col + blurCol;
            if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                pixR += in[3*(curRow * w + curCol)];
                pixG += in[3*(curRow * w + curCol) + 1];
                pixB += in[3*(curRow * w + curCol) + 2];
                pixels++;
            }
        }

        out[3*(Row * w + Col)]     = (unsigned char)(pixR / pixels);
        out[3*(Row * w + Col) + 1] = (unsigned char)(pixG / pixels);
        out[3*(Row * w + Col) + 2] = (unsigned char)(pixB / pixels);
    }
}

// Funzione per leggere PNG RGB usando lodepng
unsigned char* readPNG(const char* filename, unsigned& width, unsigned& height) {
    vector<unsigned char> imageRGBA;
    unsigned error = lodepng::decode(imageRGBA, width, height, filename);
    if (error) {
        cerr << "Errore nel decodificare PNG: " << lodepng_error_text(error) << endl;
        return nullptr;
    }

    // Converti RGBA in RGB
    unsigned char* imageRGB = (unsigned char*)malloc(width * height * 3);
    for (unsigned i = 0, j = 0; i < imageRGBA.size(); i += 4, j += 3) {
        imageRGB[j]     = imageRGBA[i];
        imageRGB[j + 1] = imageRGBA[i + 1];
        imageRGB[j + 2] = imageRGBA[i + 2];
    }
    return imageRGB;
}

int main() {
    const char* inputFile = "input.png";
    const char* outputFile = "blurred.png";

    unsigned width, height;
    unsigned char* h_image = readPNG(inputFile, width, height);
    if (!h_image) return 1;

    // Dimensione buffer RGB
    size_t imageSize = width * height * 3 * sizeof(unsigned char);

    // Allocazione GPU
    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, imageSize);
    cudaMalloc((void**)&d_out, imageSize);

    // Copia immagine su GPU
    cudaMemcpy(d_in, h_image, imageSize, cudaMemcpyHostToDevice);

    // Configura blocchi e griglia
    dim3 block(16,16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    // Lancia kernel blur
    blurKernel<<<grid, block>>>(d_out, d_in, width, height);
    cudaDeviceSynchronize();

    // Copia risultato su CPU
    unsigned char* h_blur = (unsigned char*)malloc(imageSize);
    cudaMemcpy(h_blur, d_out, imageSize, cudaMemcpyDeviceToHost);

    // Salva PNG RGB
    unsigned error = lodepng::encode(outputFile, h_blur, width, height, LCT_RGB);
    if (error) cerr << "Errore nel salvare PNG: " << lodepng_error_text(error) << endl;
    else cout << "Immagine blurrata salvata in " << outputFile << endl;

    // Cleanup
    free(h_image);
    free(h_blur);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
