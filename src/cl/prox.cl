#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

void swap(float *a, float *b) {
    float tmp;
    tmp = *b;
    *b = *a;
    *a = tmp;
}

void bubbleSort(__global float *prox, unsigned int blockSize) {
    const unsigned int index = get_global_id(0);
    for (unsigned int i = 0; i < blockSize - 1; i++) {
        for (unsigned int j = 0; j < blockSize - i - 1; j++) {
            if (prox[index * blockSize + j] > prox[index * blockSize + j + 1]) {
                float tmp;
                tmp = prox[index * blockSize + j];
                prox[index * blockSize + j] = prox[index * blockSize + j + 1];
                prox[index * blockSize + j + 1] = tmp;
            }
        }
    }
}

__kernel void prox(__global float *observations,
                   __global float *histogram,
                   __global float *image,
                   __global float *prox,
                   unsigned int width, unsigned int height,
                   unsigned int n, unsigned int amountOfObservation, float tau, float lambda_data) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    const unsigned int imageSize = width * height;
    const unsigned int blockSize = amountOfObservation * 2;
    if (index <= n) {
        for (unsigned int i = 0; i < blockSize / 2; i++) {
            prox[i + blockSize * index] = observations[index + imageSize * i];
        }
        for (unsigned int i = blockSize / 2; i < blockSize; i++) {
            prox[i + blockSize * index] = image[index] + tau*lambda_data*histogram[index + imageSize * i / 2];
        }
        bubbleSort(prox, blockSize);
        image[index] = (prox[blockSize / 2 + index * blockSize] + prox[blockSize / 2 + index * blockSize - 1]) / 2;
    }
}
