#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6


void sum(__global float *a, __global float *b, unsigned int size,
         unsigned int dim) { //potential speed up - being cache friendly by summing only sequence of elements with size of dimension
    const unsigned int index = get_global_id(0);
    for (unsigned int i = 0; i < dim; i++) {
        a[index + i * size] += b[index + i * size];
    }
}

__kernel void sumOfMatrix(__global float *a,
                          __global float *b,
                          unsigned int width, unsigned int height, unsigned int dim,
                          unsigned int n) {//n - image size
    const unsigned int index = get_global_id(0);
    if (index < n) {
        sum(a, b, width * height, dim);
    }
}