#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void calculateTranspondedGradient(__global float *gradient, __global float *transpondedGradient, unsigned int width,
                                  unsigned int height) {
    const unsigned int index = get_global_id(0);
    unsigned int imageSize = width * height;
    transpondedGradient[index] = (-gradient[index]);
    transpondedGradient[index] -= gradient[index + imageSize];
    if (index % width != (width - 1)) {
        transpondedGradient[index + 1] += gradient[index];
    }
    if (index / width != (height - 1)) {
        transpondedGradient[index + width] += gradient[index + imageSize];
    }

}

__kernel void transpondedGradient(__global float *gradient,
                       __global float *transpondedGradient,
                       unsigned int width, unsigned int height,
                       unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index <= n) {
        calculateTranspondedGradient(gradient, transpondedGradient, width, height);
    }
}
