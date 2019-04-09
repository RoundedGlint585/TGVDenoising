#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void calculateHist(__global float *observations,
                            __global float *histogram,
                            unsigned int width, unsigned int height,
                            unsigned int n, unsigned int amountOfObservation) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    const unsigned int imageSize = width * height;
    if (index <= n) {
        for (unsigned int i = 0; i < amountOfObservation; i++) {
            histogram[index + i * imageSize] = 0;
            for (unsigned int j = 0; j < amountOfObservation; j++) {
                if (observations[index + i * imageSize] > observations[index + j * imageSize]) {
                    histogram[index + i * imageSize]++;
                } else {
                    histogram[index + i * imageSize]--;
                }
            }
        }
    }
}
