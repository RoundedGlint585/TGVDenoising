#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void anorm( const __global float *matrix,
                      __global float *result,
                      unsigned int width, unsigned int height,
                      unsigned int n, unsigned int dim, float r) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    const unsigned int imageSize = width * height;
    const float eps = 0.001f;
    if (index < n) {
        float normalized = 0.f;
        for (unsigned int i = 0; i < dim; i++) {
            normalized += (matrix[index + imageSize * i]*matrix[index + imageSize * i]);
        }
        result[index] = sqrt(normalized)/r;
//        if(result[index] < eps){
//            result[index] = 1.f;
//        }
    }
}