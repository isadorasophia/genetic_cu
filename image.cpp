#include "image.h"

/* cuda header */
#ifdef CUDA
#include "image_op.h"
#endif

#include <iostream>

/* -- thrust library -- */
// #include <thrust/sort.h>
// #include <thrust/reduce.h>

/* Find min. and max. value from a vector! */
// inline void find_min_max(thrust::device_vector<int> &dev_vec, int *min, int *max){
//     thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> tuple;

//     tuple = thrust::minmax_element(dev_vec.begin(), dev_vec.end());

//     *min = *(tuple.first);
//     *max = *(tuple.second);
// }

// /* Find polygon, still figuring out way to receive vx and vy */
// struct find_poly
// {
//     __host__ __device__
//     float operator() (thrust::device_vector vx, thrust::device_vector vy)
//     {
//         return x * x;
//     }
// };

uint64_t Image::
Compare(const Image &src) const {
    uint64_t error = 0;
    const unsigned char *s = (const unsigned char *)src.surface_->pixels;
    const unsigned char *d = (const unsigned char *)surface_->pixels;
    uint16_t pitch = surface_->pitch;
    uint16_t bpp = surface_->format->BytesPerPixel;
    int h = surface_->h;

#if 0
    bpp *= 2;
    pitch *= 2;
#endif

#ifdef CUDA
    /* --- cuda implementation --- */
    int w = surface_->w;

    error = compare_cu(s, d, pitch, bpp, h, w);
#else
    /* --- old code --- */
    while (--h) {
        const unsigned char *dline = d, *sline = s;
        int w = surface_->w;

        while (--w) { // expect pixels as RGBA!
            int er = (int)(dline[1]) - (int)(sline[1]);
            int eg = (int)(dline[2]) - (int)(sline[2]);
            int eb = (int)(dline[3]) - (int)(sline[3]);

            error += ((er * er) + (eb * eb) + (eg * eg));

            dline += bpp;
            sline += bpp;
        }

        d += pitch;
        s += pitch;
    }
#endif

    return error;
}

void Image::
Polygon(const Sint16 * vx, const Sint16 * vy, int n, Uint32 color)
{
    int i;
    int y, xa, xb;
    int miny, maxy;
    int x1, y1;
    int x2, y2;
    int ind1, ind2;
    int ints;
    uint8_t alpha = color & 0x000000ff;
    uint32_t R = color >> 24;
    uint32_t G = (color >> 16) & 0xff;
    uint32_t B = (color >> 8) & 0xff;

    /* -- old code -- */
    int gfxPrimitivesPolyInts[n];

    /* Initialize memory */
    // thrust::host_vector<int>   gfxPrimitivesPolyInts_h(n); // used by user (host)
    // thrust::device_vector<int> gfxPrimitivesPolyInts_d(n); // used by GPU (device)

    R *= alpha;
    G *= alpha;
    B *= alpha;

    alpha = 255 - alpha;
    /*
     * Check visibility of clipping rectangle
     */
    if ((surface_->clip_rect.w==0) || (surface_->clip_rect.h==0)) return;

    /*
     * Sanity check number of edges
     */
    if (n < 3) return;

    /*
     * Determine Y maxima 
     */
    /* Deal with memory */
    // thrust::host_vector<int>   vx_h(n); // used by user (host)
    // thrust::device_vector<int> vx_d(n); // used by GPU (device)

    // thrust::host_vector<int>   vy_h(n); // used by user (host)
    // thrust::device_vector<int> vy_d(n); // used by GPU (device)
    
    // thrust::copy(vx, vx + n, vx_h.begin());
    // thrust::copy(vy, vy + n, vy_h.begin());

    /* Find min. and max. */
    // vx_d = vx_h;
    // find_min_max(vx_d, &miny, &maxy);

    /* -- old code -- */
    miny = vy[0];
    maxy = vy[0];
    for (i = 1; (i < n); i++) {
        if (vy[i] < miny) {
            miny = vy[i];
        } else if (vy[i] > maxy) {
            maxy = vy[i];
        }
    }

    /*
     * [HOT AREA] Draw, scanning y 
     */
    for (y = miny; (y <= maxy); y++) {
        /* -- find a way to apply transform here -- */
        for (i = 0, ints = 0; (i < n); i++) {
            if (!i) {
                ind1 = n - 1;
                ind2 = 0;
            } else {
                ind1 = i - 1;
                ind2 = i;
            }
            y1 = vy[ind1];
            y2 = vy[ind2];
            if (y1 < y2) {
                x1 = vx[ind1];
                x2 = vx[ind2];
            } else if (y1 > y2) {
                y2 = vy[ind1];
                y1 = vy[ind2];
                x2 = vx[ind1];
                x1 = vx[ind2];
            } else {
                continue; 
            }

            if ( ((y >= y1) && (y < y2)) || ((y == maxy) && (y > y1) && (y <= y2)) ) {
                gfxPrimitivesPolyInts[ints++] = ((65536 * (y - y1)) / (y2 - y1)) * (x2 - x1) + (65536 * x1);
            }
        }

        // copy host to device
        // gfxPrimitivesPolyInts_d = gfxPrimitivesPolyInts_h;

        // sort it, given an array and its size
        // thrust::sort(gfxPrimitivesPolyInts.begin(), gfxPrimitivesPolyInts.end());

        /* -- old code -- */
        qsort(gfxPrimitivesPolyInts, ints, sizeof(int), CompareInt);

        /* -- print into the screen -- */
        for (i = 0; (i < ints); i += 2) {
            xa = gfxPrimitivesPolyInts[i] + 1;
            xa = (xa >> 16) + ((xa & 32768) >> 15);
            xb = gfxPrimitivesPolyInts[i+1] - 1;
            xb = (xb >> 16) + ((xb & 32768) >> 15);

            HLineAlpha(xa, xb, y, R, G, B, alpha);
        }
    }
}

void Image::
HLineAlpha(Sint16 x1, Sint16 x2, Sint16 y, Uint32 r, Uint32 g, Uint32 b, Uint8 alpha) {
        Uint32 R, G, B;
        Uint32 *row;

        row = (Uint32 *) surface_->pixels + y * surface_->pitch / 4 + x1;
        x2 -= x1;

        for (; x2 >= 0; x2--, row++) {
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            R = ((unsigned char *)row)[0];
            G = ((unsigned char *)row)[1];
            B = ((unsigned char *)row)[2];
#else
            R = ((unsigned char *)row)[3];
            G = ((unsigned char *)row)[2];
            B = ((unsigned char *)row)[1];
#endif           
            R *= alpha;
            G *= alpha;
            B *= alpha;
            R += r;
            G += g;
            B += b;

            // every component here has to be divided /256, but we can avoid this
            *row = ((R & 0xff00) << 16) | ((G & 0xff00) << 8) | (B & 0xff00) | 0xff;
        }
    }

