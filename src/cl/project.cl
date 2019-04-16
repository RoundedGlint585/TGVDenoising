#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6


__kernel void project(__global float *matrix,
                      unsigned int width, unsigned int height,
                      unsigned int n, unsigned int dim, float r) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    const unsigned int imageSize = width * height;
    const float eps = 0.001;
    if (index < n) {
        float normalized = 0;
        for (int i = 0; i < dim; i++) {
            normalized += matrix[index + imageSize * i] * matrix[index + imageSize * i];
        }
        normalized = sqrt(normalized);
        normalized /= r;
        if (normalized < eps) {
            normalized = 1.f;
        }
        for (int i = 0; i < dim; i++) {
            matrix[index + imageSize * i] /= normalized;
        }
    }
}