#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6



__kernel void clear(__global float *a,
                   unsigned int width, unsigned int height, unsigned int dim,
                   unsigned int n) {//n - image size
    const unsigned int index = get_global_id(0);
    if (index < n) {
        a[index] = 0.f;
    }

}