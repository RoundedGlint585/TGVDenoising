#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void project(__global float *matrix, __global float *normed,
                      unsigned int width, unsigned int height,
                      unsigned int n, unsigned int dim) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    const unsigned int imageSize = width * height;
    if (index < n) {
        for (unsigned int i = 0; i < dim; i++) {
            if(normed[index] < 0.00001f){
                normed[index] = 1.f;
            }
            matrix[index + imageSize * i] = matrix[index + imageSize * i] /  normed[index];
        }
    }
}