#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6


void mul(__global float *from, float k, unsigned int size,
         unsigned int dim) { //potential speed up - being cache friendly by summing only sequence of elements with size of dimension
    const unsigned int index = get_global_id(0);
    for (unsigned int i = 0; i < dim; i++) {
        from[index + i * size] *= k;
    }
}

__kernel void mulMatrixOnConstant(__global float *from,
                                  float k,
                                  unsigned int width, unsigned int height, unsigned int dim,
                                  unsigned int n) {//n - image size
    const unsigned int index = get_global_id(0);
    if (index <= n) {
        mul(from, k, width * height, dim);
    }
}