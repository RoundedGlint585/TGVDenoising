#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

///layout for epsilon: dx_01...dx_n1 dy_01...dy_n1 dx_02...dx_n2 dy_02...dy_n2
void calculateEpsilon(__global float *gradient, __global float *epsilon, unsigned int width, unsigned int height) {
    const unsigned int index = get_global_id(0);

    unsigned int imageSize = width * height;
    if (index % width == (width - 1)) {
        epsilon[index] = 0;
        epsilon[index + 2 * imageSize] = 0;
    } else {
        epsilon[index] = gradient[index + 1] - gradient[index];
        epsilon[index + 2 * imageSize] = gradient[index + 1 + imageSize] - gradient[index + imageSize];
    }
    if (index / width == height - 1) {
        epsilon[index + imageSize] = 0;
        epsilon[index + 3 * imageSize] = 0;
    } else {
        epsilon[index + imageSize] = gradient[index + width] - gradient[index];
        epsilon[index + 3 * imageSize] = gradient[index + width +imageSize] - gradient[index+imageSize];
    }
}

__kernel void epsilon(__global float *gradient,
                      __global float *epsilon,
                      unsigned int width, unsigned int height,
                      unsigned int n) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    if (index < n) {
        calculateEpsilon(gradient, epsilon, width, height);
    }


}