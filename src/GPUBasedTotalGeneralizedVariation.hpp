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
class GPUBasedTGV {
public:
    GPUBasedTGV(size_t argc, char **argv);

    void init();

    void start(size_t iterations);


private:
    std::tuple<size_t, size_t, int, int, std::vector<float>, std::vector<float>> loadImages(std::string_view path);

    void loadData(size_t name, const std::vector<float> &data);

    void reserveDataN(size_t name, size_t amount);

    std::vector<float> getBuffer(size_t name) const;


    enum index {
        image, v, p, q, observations, transpondedGradient, transpondedEpsilon
    };
    gpu::Device device;
    gpu::Context context;
    size_t memsize;
    std::array<std::pair<size_t, gpu::gpu_mem_32f>, 6> memoryBuffers; //0 - image(u), 1 - v , 2 - p, 3- q, 4-images, 5 - transpondedGradient, 6 - transpondedEpsilon
    ocl::Kernel tgvEpsilonKernel, tgvGradientKernel, tgvCopyKernel;
    size_t width = 0;
    size_t height = 0;
    size_t amountOfObservation = 0;
};

