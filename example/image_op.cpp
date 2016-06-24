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