//
// Created by roundedglint585 on 4/2/19.
//

#include "GPUBasedTotalGeneralizedVariation.hpp"
#include <filesystem>

GPUBasedTGV::GPUBasedTGV(size_t argc, char **argv) {
    device = gpu::chooseGPUDevice(argc, argv);
    context.init(device.device_id_opencl);
    context.activate();
    tgvEpsilonKernel = ocl::Kernel(epsilon_kernel, epsilon_kernel_length, "epsilon");
    tgvGradientKernel = ocl::Kernel(gradient_kernel, gradient_kernel_length, "gradient");
    tgvCopyKernel = ocl::Kernel(copy_kernel, copy_kernel_length, "copy");
    tgvEpsilonKernel.compile();
    tgvGradientKernel.compile();
    memsize = device.mem_size;
}

void GPUBasedTGV::init() {
    auto resultOfLoading = loadImages("data");
    loadData(observations, std::get<5>(resultOfLoading));
    loadData(image, std::get<4>(resultOfLoading));
    width = std::get<2>(resultOfLoading);
    height = std::get<3>(resultOfLoading);
    amountOfObservation = std::get<5>(resultOfLoading).size();
    size_t sizeOfImage = height * width;
    reserveDataN(v, sizeOfImage * 2);
    reserveDataN(p, sizeOfImage * 2);
    reserveDataN(q, sizeOfImage * 4);
    reserveDataN(transpondedGradient, sizeOfImage);
    std::cout << "Memory allocated and reserved" << std::endl;
}

std::tuple<size_t, size_t, int, int, std::vector<float>, std::vector<float>>
GPUBasedTGV::loadImages(std::string_view path) {
    using namespace std::filesystem;
    size_t totalSize = 0;
    size_t amountOfImages = 0;
    int width, height;
    int channels;
    std::vector<float> observations;
    std::vector<float> image;
    for (auto &p: directory_iterator(path)) {
        std::cout << "loading image: " << p << std::endl;
        std::string name = p.path();
        unsigned char *bytes = stbi_load(name.c_str(),
                                         &width,
                                         &height,
                                         &channels,
                                         STBI_grey);
        if (amountOfImages != 0) {
            for (size_t i = 0; i < width * height; i++) {
                observations.emplace_back((float) bytes[i]);
            }
        } else {
            for (size_t i = 0; i < width * height; i++) {
                image.emplace_back((float) bytes[i]);
            }
        }
        amountOfImages++;
        totalSize += width * height;
    }
    return std::make_tuple(totalSize, amountOfImages, width, height, image, observations);
}

void GPUBasedTGV::reserveDataN(size_t name, size_t amount) {
    memoryBuffers[name].second.resizeN(amount);
    memoryBuffers[name].first = amount;
}

void GPUBasedTGV::loadData(size_t name, const std::vector<float> &data) {
    memoryBuffers[name].second.resizeN(data.size());
    memoryBuffers[name].first = data.size();
    memoryBuffers[name].second.writeN(data.data(), data.size());
}

std::vector<float> GPUBasedTGV::getBuffer(size_t name) const {
    auto buffer = memoryBuffers[name];
    std::vector<float> result(buffer.first, 0.f);
    buffer.second.readN(result.data(), buffer.first);
    return result;
}

void GPUBasedTGV::start(size_t iterations) {
    unsigned int workGroupSize = 128;
    std::cout << "Image size: " << memoryBuffers[image].first << std::endl;
    unsigned int global_work_size = (memoryBuffers[image].first + workGroupSize - 1) / workGroupSize * workGroupSize;
    std::cout << "Global work size: " << global_work_size << std::endl;
    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, global_work_size), memoryBuffers[image].second,
                      memoryBuffers[v].second, (unsigned int)
                              width, (unsigned int) height, (unsigned int) memoryBuffers[image].first);


}

