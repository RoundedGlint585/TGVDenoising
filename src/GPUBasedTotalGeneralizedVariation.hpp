//
// Created by roundedglint585 on 4/2/19.
//
#pragma once

#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "StbInterfaceProxy.hpp"
#include "commonKernels.hpp"

class GPUBasedTGV {
public:
    GPUBasedTGV(std::size_t index);

    void init(size_t amountOfImages, size_t width, size_t height, const std::vector<float> &image,
              const std::vector<float> &observations);

    void init(const std::vector<float> &observations, size_t height, size_t width);

    void iteration(float tau, float lambda_tv, float lambda_tgv, float lambda_data);

    std::vector<float> getImage() const;

    size_t getHeight() const;

    size_t getWidth() const;

private:

    void initKernels();

    void reserveDataN(size_t name, size_t amount);

    std::vector<float> getBuffer(size_t name) const;

    void loadData(size_t name, const std::vector<float> &data);

    void calculateImageDual(float tau_u, float lambda_tv, float tau, float lambda_data);

    void calculateVDual(float tau_v, float lambda_tgv, float lambda_tv);

    void calculatePDual(float tau_p, float lambda_tv);

    void calculateQDual(float tau_q, float lambda_tgv);


    enum bufIndex {
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
    std::array<std::pair<size_t, gpu::gpu_mem_32f>, 13> memoryBuffers; //mapped from index

    ocl::Kernel tgvEpsilonKernel, tgvGradientKernel, tgvTranspondedEpsilonKernel, tgvTranspondedGradientKernel, tgvMulMatrixOnConstantKernel,
            tgvSumOfMatrixKernel, tgvProjectKernel, tgvCopyKernel, tgvCalculateHistKernel, tgvProxKernel, tgvClearKernel, tgvAnormKernel;
    size_t width = 0;
    size_t height = 0;
    size_t amountOfObservation = 0;
    unsigned int workGroupSize;
    unsigned int globalWorkSize;

};

