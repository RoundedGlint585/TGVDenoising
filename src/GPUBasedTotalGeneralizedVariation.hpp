//
// Created by roundedglint585 on 4/2/19.
//
#pragma once

#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <unordered_map>
#include <stb_image.h>
#include <stb_image_write.h>
#include "cl/epsilon.h"
#include "cl/gradient.h"
#include "cl/copy.h"
#include "cl/mulMatrixOnConstant.h"
#include "cl/project.h"
#include "cl/sumOfMatrix.h"
#include "cl/transpondedGradient.h"
#include "cl/transpondedEpsilon.h"
#include "cl/calculateHist.h"
#include "cl/prox.h"
#include "cl/clear.h"
#include "cl/anorm.h"

class GPUBasedTGV {
public:
    GPUBasedTGV(size_t argc, char **argv, size_t amountOfImagesGPU);

    void init();

    void start(size_t iterations, float tau, float lambda_tv, float lambda_tgv, float lambda_data);

    void writeResult(const std::string &name);

    std::vector<float> getImage();

private:
    std::tuple<size_t, size_t, int, int, std::vector<float>, std::vector<float>> loadImages(std::string_view path);

    void initKernels();

    void loadData(size_t name, const std::vector<float> &data);

    void reserveDataN(size_t name, size_t amount);

    std::vector<float> getBuffer(size_t name) const;

    void iteration(float tau, float lambda_tv, float lambda_tgv, float lambda_data, unsigned int workGroupSize,
                   unsigned int globalWorkSize);

    void calculateImageDual(float tau_u, float lambda_tv, float tau, float lambda_data, unsigned int workGroupSize,
                            unsigned int globalWorkSize);

    void calculateVDual(float tau_v, float lambda_tgv, float lambda_tv, unsigned int workGroupSize,
                        unsigned int globalWorkSize);

    void calculatePDual(float tau_p, float lambda_tv, unsigned int workGroupSize,
                        unsigned int globalWorkSize);

    void calculateQDual(float tau_q, float lambda_tgv, unsigned int workGroupSize,
                        unsigned int globalWorkSize);

    enum index {
        image,
        v,
        p,
        q,
        observations,
        imageDual,
        vDual,
        pDual,
        qDual,
        transpondedGradient,
        transpondedEpsilon,
        histogram,
        prox
    };
    gpu::Device device;
    gpu::Context context;
    size_t memsize;
    std::array<std::pair<size_t, gpu::gpu_mem_32f>, 13> memoryBuffers; //mapped from index
    ocl::Kernel tgvEpsilonKernel, tgvGradientKernel, tgvTranspondedEpsilonKernel, tgvTranspondedGradientKernel, tgvMulMatrixOnConstantKernel,
            tgvSumOfMatrixKernel, tgvProjectKernel, tgvCopyKernel, tgvCalculateHistKernel, tgvProxKernel, tgvClearKernel, tgvAnormKernel;
    size_t width = 0;
    size_t height = 0;
    size_t amountOfObservation = 0;
    size_t amountOfImagesToGPU = 0;
};
