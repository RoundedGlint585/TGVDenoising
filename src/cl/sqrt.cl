#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void sqrtCalc(__global float *from,
                   unsigned int n) {//n - image size
    const unsigned int index = get_global_id(0);
    if (index < n) {
        from[index] = half_sqrt(from[index]);
    }

}