#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

///Layout for gradient: dx_0...dx_n dy_0...dy_n


void calculateGradient(__global float *image, __global float *v, unsigned int width, unsigned int height) {
    const unsigned int index = get_global_id(0);

    unsigned int imageSize = width * height;
    if (index % width == (width - 1)) { //border on width
        v[index] = 0;
    } else {
        v[index] = image[index] - image[index - 1];
    }
    if (index / width == height - 1) { //border on height
        v[index + imageSize] = 0;
    } else {
        v[index + imageSize] = image[index] - image[index + width];
    }
}

__kernel void gradient(__global float *image,
                       __global float *gradient,
                       unsigned int width, unsigned int height,
                       unsigned int n) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    if (index <= n) {
        calculateGradient(image, gradient, width, height);
    }
}
