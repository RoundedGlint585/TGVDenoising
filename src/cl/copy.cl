#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void copyFromTo(__global float *v, __global float *p, unsigned int size, unsigned int dim) {
    const unsigned int index = get_global_id(0);
    for (unsigned int i = 0; i < dim; i++) {
        v[index + i * size] = p[index + i * size];
    }

}

__kernel void copy(__global float *from,
                   __global float *to,
                   unsigned int width, unsigned int height, unsigned int dim,
                   unsigned int n) {//n - image size
    const unsigned int index = get_global_id(0);
    if (index <= n) {
        copyFromTo(from, to, width * height, dim);
    }

}