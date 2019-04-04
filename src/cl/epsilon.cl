#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void calculateEpsilon(__global float *gradient, __global float *epsilon, unsigned int width, unsigned int height) {
    const unsigned int index = get_global_id(0);

    unsigned int imageSize = width * height;
    if (index % width == (width - 1)) {
        printf("%d\n", index);
        epsilon[index] = 0;
        epsilon[index + 2 * imageSize] = 0;
    } else {
        epsilon[index] = gradient[index] - gradient[index - 1];
        epsilon[index + 2 * imageSize] = gradient[index + imageSize] - gradient[index + imageSize - 1];
    }
    if (index / width == height - 1) {
        epsilon[index + imageSize] = 0;
        epsilon[index + 3 * imageSize] = 0;
    } else {
        epsilon[index + imageSize] = gradient[index] - gradient[index + width];
        epsilon[index + 2 * imageSize] = gradient[index + imageSize] - gradient[index + imageSize + width];
    }
}

__kernel void epsilon(__global float *gradient,
                  __global float *epsilon,
                  unsigned int width, unsigned int height,
                  unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index <= n) {
        calculateEpsilon(gradient, epsilon, width, height);
    }


}