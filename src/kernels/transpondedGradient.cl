#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void calculateTranspondedGradient(__global float *gradient, __global float *transpondedGradient, unsigned int width,
                                  unsigned int height) {
    const int index = get_global_id(0);
    int imageSize = width * height;
    transpondedGradient[index] = -gradient[index] - gradient[index + imageSize];
    if (index % width != 0) {
        transpondedGradient[index] += gradient[index - 1];
    }
    if (index / width != 0) {
        transpondedGradient[index] += gradient[index + imageSize - width];
    }


}

__kernel void transpondedGradient(__global float *gradient,
                                  __global float *transpondedGradient,
                                  unsigned int width, unsigned int height,
                                  unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index < n) {
        calculateTranspondedGradient(gradient, transpondedGradient, width, height);
    }
}