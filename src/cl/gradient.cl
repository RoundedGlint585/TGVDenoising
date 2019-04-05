#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void calculateGradient(__global float *image, __global float *v, unsigned int width, unsigned int height) {
    const unsigned int index = get_global_id(0);

    unsigned int imageSize = width * height;
    if (index % width == (width - 1)) {
        v[index] = 0;
    } else {
        v[index] = image[index] - image[index - 1];
    }
    if (index / width == height - 1) {
        v[index + imageSize] = 0;
    } else {
        v[index + imageSize] = image[index] - image[index + width];
    }
}


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

__kernel void gradient(__global float *image,
                       __global float *gradient,
                       unsigned int width, unsigned int height,
                       unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index <= n) {
        calculateGradient(image, gradient, width, height);
        //copyFromTo(v, p, width * height, 2);
        //calculateTranspondedGradient(v, transpondedGradient, width, height);
    }


}
