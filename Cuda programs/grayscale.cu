#include "lodepng.h"
#include <iostream>
#include <vector>
using namespace std;

#define CHANNELS 3
__global__ void colorToGrey(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
        int greyOffset = Row*width + Col; // get 1D coordinate for the grayscale image
        int rgbOffset = greyOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset ]; // red value for pixel
        unsigned char g = Pin[rgbOffset + 1]; // green value for pixel
        unsigned char b = Pin[rgbOffset + 2]; // blue value for pixel
        Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}


unsigned char* readPNG(const char* filename, unsigned& width, unsigned& height) {
    vector<unsigned char> imageRGBA;

    // Decodifica PNG in RGBA
    unsigned error = lodepng::decode(imageRGBA, width, height, filename);

    if (error) {
        cerr << "Errore nel decodificare PNG: " << lodepng_error_text(error) << endl;
        return nullptr;
    }

    cout << "Immagine caricata: " << width << "x" << height << " (RGBA)" << endl;

    // Converti da RGBA a RGB (elimina alpha)
    unsigned char* imageRGB = (unsigned char*)malloc(width * height * 3);
    for (unsigned i = 0, j = 0; i < imageRGBA.size(); i += 4, j += 3) {
        imageRGB[j]     = imageRGBA[i];     // R
        imageRGB[j + 1] = imageRGBA[i + 1]; // G
        imageRGB[j + 2] = imageRGBA[i + 2]; // B
    }

    return imageRGB; // restituisce buffer RGB contiguo
}


int main() {
    unsigned width, height;
    unsigned char* image = readPNG("input.png", width, height);
    
    int rgbSize = width * height * 3 * sizeof(unsigned char);
    int greySize = width * height * sizeof(unsigned char);
    unsigned char *Pin_dev, *Pout_dev;
    unsigned char* grey_host = (unsigned char*)malloc(width * height);

    cudaMalloc((void**)&Pin_dev ,rgbSize);
    cudaMalloc((void**)&Pout_dev ,rgbSize);

    cudaMemcpy(Pin_dev, image, rgbSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    colorToGrey<<<grid, block>>>(Pout_dev, Pin_dev, width, height);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Errore kernel: " << cudaGetErrorString(err) << endl;
    }

    cudaMemcpy(grey_host, Pout_dev, greySize, cudaMemcpyDeviceToHost);

    unsigned error = lodepng::encode("output.png", grey_host, width, height, LCT_GREY);
    if (error) {
        cerr << "Errore salvataggio PNG: " << lodepng_error_text(error) << endl;
    }

    cudaFree(Pout_dev);
    cudaFree(Pin_dev);
    free(image);
    free(grey_host);

    return 0;
}
