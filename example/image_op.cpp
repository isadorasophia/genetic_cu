/* -- thrust library -- */
#include "image_op.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <thrust/sort.h>
#include <thrust/reduce.h>

#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#define TILE_WIDTH 16
#define HEIGHT 1080

#define uint64_t unsigned long int

__global__ void find_error(const unsigned char *s, const unsigned char *d, uint64_t *error, uint16_t pitch, uint16_t bpp, int h, int w) {
    extern __shared__ uint64_t d_errors[]; // shared error data

    unsigned int row = blockDim.x * blockIdx.x + blockIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + blockIdx.y;

    /* Initialize */
    if (row < h && col == 0) {
        d_errors[row] = 0;
    }

    __syncthreads();

    /* Get correct position */
    const unsigned char *dline = d, *sline = s;

    dline += pitch * row + bpp * col;
    sline += pitch * row + bpp * col;

    /* Estimate error */
    int er = (int)(dline[1]) - (int)(sline[1]);
    int eg = (int)(dline[2]) - (int)(sline[2]);
    int eb = (int)(dline[3]) - (int)(sline[3]);

    uint64_t f_e = (er * er) + (eb * eb) + (eg * eg);

    atomicAdd(&d_errors[row], f_e);

    __syncthreads();

    /* Add final result */
    if (row < h && col == 0) {
        atomicAdd(error, d_errors[row]);
    }
}

uint64_t compare_cu(const unsigned char *s, const unsigned char *d, uint16_t pitch, uint16_t bpp, int h, int w) {
    uint64_t error = 0;
    unsigned int size = h * w;      // total of pixels

    /* --- initialize data --- */
    unsigned char *d_s, *d_d; // device arrays
    uint64_t* d_error;        // device error

    /* --- create buffer --- */
    cudaMalloc((void **) &d_s, sizeof(const unsigned char) * size);
    cudaMalloc((void **) &d_d, sizeof(const unsigned char) * size);
    cudaMalloc((void **) &d_error, sizeof(uint64_t));

    /* --- offload sender --- */
    cudaMemcpy(d_s, s, sizeof(const unsigned char) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, sizeof(const unsigned char) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_error, &error, sizeof(uint64_t), cudaMemcpyHostToDevice);

    /* --- kernel code --- */
    /* get dimensions */
    dim3 dimGrid(ceil((float)w/TILE_WIDTH), ceil((float)h/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    find_error <<< dimGrid, dimBlock, h >>> (d_s, d_d, d_error, pitch, bpp, h, w);
    
    /* --- offload receiver --- */
    cudaMemcpy(&error, d_error, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    /* --- clean up --- */
    cudaFree(d_s);
    cudaFree(d_d);
    cudaFree(d_error);

    return error;
}

void sort(int* array, int size) {
    thrust::host_vector<int> a_h(size);
    thrust::device_vector<int> a_d;

    thrust::copy(array, array + size, a_h.begin());

    a_d = a_h;

    thrust::sort(a_d.begin(), a_d.end());

    thrust::copy(a_d.begin(), a_d.end(), array);
}
