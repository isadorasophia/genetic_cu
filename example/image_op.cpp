/* -- thrust library -- */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <thrust/sort.h>
#include <thrust/reduce.h>

#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

__global__ void convolution(const unsigned char *s, const unsigned char *d, uint64_t *error, uint16_t pitch, uint16_t bpp, int h, int w) {
            
            int i = blockDim.x * blockIdx.x  + threadIdx.x;
            int dlineIterator = i%w;
            int dIterator = (i/w)%h;
            d += pitch*dIterator;
            s += pitch*dIterator;


            const unsigned char *dline = d;
            const unsigned char *sline = s;

            dline += bpp*dlineIterator;
            sline += bpp*dlineIterator;
           
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            int er = (int)(dline[0]) - (int)(sline[0]);
            int eg = (int)(dline[1]) - (int)(sline[1]);
            int eb = (int)(dline[2]) - (int)(sline[2]);
#else
            int er = (int)(dline[1]) - (int)(sline[1]);
            int eg = (int)(dline[2]) - (int)(sline[2]);
            int eb = (int)(dline[3]) - (int)(sline[3]);
#endif

            atomicAdd(&(error[0]), ((er * er) + (eb * eb) + (eg * eg)));
            //error[0] += ((er * er) + (eb * eb) + (eg * eg));
}

uint64_t compare(const Image &src) {
    const unsigned char *s = (const unsigned char *)src.surface_->pixels;
    const unsigned char *d = (const unsigned char *)surface_->pixels;

    const unsigned char *d_s, *d_d;
    int h = surface_->h;
    int w = surface_->w;
    /* Nao tenho certeza do sizeof()*h*w!!! */
    cudaMalloc(d_s, sizeof(const unsigned char)*h*w); /*COMO PEGAR O TAMANHO DELES PARA DAR MALLOC???? */
    cudaMalloc(d_d, sizeof(const unsigned char)*h*w); /*COMO PEGAR O TAMANHO DELES PARA DAR MALLOC???? */

    cudaMemcpy(d_s, s, sizeof(const unsigned char)*h*w,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, sizeof(const unsigned char)*h*w,  cudaMemcpyHostToDevice);

    uint64_t *error = (uint64_t *)malloc(sizeof(uint64_t));
    error[0] = 0;

    uint64_t *d_error;
    cudaMalloc(d_error, sizeof(uint64_t));
    cudaMemcpy(d_error, error, sizeof(uint64_t),  cudaMemcpyHostToDevice);
    uint16_t pitch = surface_->pitch;
    uint16_t bpp = surface_->format->BytesPerPixel;
#if 0
    bpp *= 2;
    pitch *= 2;
#endif

    int iterations = h*w;

    int blockDimension = 128;
    int threads = iterations;
    int threadsPerBlock = ceil(threads/(float)blockDimension);

    calculaErro<<<blockDimension, threadsPerBlock>>>(d_s, d_d, d_error, pitch, bpp, h, w);
    
    cudaMemcpy(error, d_error, sizeof(uint64_t), cudaMemcpyDeviceToHost);           
    cudaFree(d_s);
    cudaFree(d_d);
    cudaFree(d_error);
    return error[0];
}

// void Polygon_cuda(const int16_t* vx, const int16_t* vy, int n, uint32_t color) {
//     return;
// }

void sort (int* array, int size) {
    thrust::host_vector<int> a_h(size);
    thrust::device_vector<int> a_d;

    thrust::copy(array, array + size, a_h.begin());

    a_d = a_h;

    thrust::sort(a_d.begin(), a_d.end());

    thrust::copy(a_d.begin(), a_d.end(), array);
}
