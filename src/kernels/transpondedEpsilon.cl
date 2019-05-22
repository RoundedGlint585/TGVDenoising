#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void calculateTranspondedEpsilon(__global float *epsilon, __global float *transpondedEpsilon, unsigned int width,
                                 unsigned int height) {
    const int index = get_global_id(0);
    int imageSize = width * height;
    transpondedEpsilon[index] = -epsilon[index] - epsilon[index + imageSize];
    transpondedEpsilon[index + imageSize] = -epsilon[index + 2 * imageSize] - epsilon[index + 3 * imageSize];
    if (index % width != 0 ) {
        transpondedEpsilon[index] = transpondedEpsilon[index] + epsilon[index-1];
        transpondedEpsilon[index + imageSize] += epsilon[index + 2 * imageSize - 1];
    }
    if (index / width != 0) {
        transpondedEpsilon[index] += epsilon[index + imageSize - width];
        transpondedEpsilon[index + imageSize] += epsilon[index + 3 * imageSize - width];
    }

}

__kernel void transpondedEpsilon(__global float *epsilon,
                                 __global float *transpondedEpsilon,
                                 unsigned int width, unsigned int height,
                                 unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index < n) {
        calculateTranspondedEpsilon(epsilon, transpondedEpsilon, width, height);
    }
}