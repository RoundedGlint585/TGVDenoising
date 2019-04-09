#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void calculateTranspondedEpsilon(__global float *epsilon, __global float *transpondedEpsilon, unsigned int width,
                                  unsigned int height) {
    const unsigned int index = get_global_id(0);
    unsigned int imageSize = width * height;
    transpondedEpsilon[index] = (-epsilon[index]);
    transpondedEpsilon[index] -= epsilon[index + imageSize];
    transpondedEpsilon[index+imageSize] = (-epsilon[index + 2*imageSize]);
    transpondedEpsilon[index+imageSize] -= epsilon[index + 3*imageSize];
    if (index % width != (width - 1)) {
        transpondedEpsilon[index + 1] += epsilon[index];
        transpondedEpsilon[index+imageSize + 1] += epsilon[index + 2*imageSize];
    }
    if (index / width != (height - 1)) {
        transpondedEpsilon[index + width] += epsilon[index + imageSize];
        transpondedEpsilon[index+ width +imageSize + 1] += epsilon[index + 3*imageSize];
    }

}

__kernel void transpondedEpsilon(__global float *epsilon,
                       __global float *transpondedEpsilon,
                       unsigned int width, unsigned int height,
                       unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index <= n) {
        calculateTranspondedEpsilon(epsilon, transpondedEpsilon, width, height);
    }
}