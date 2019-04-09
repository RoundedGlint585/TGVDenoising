//
// Created by roundedglint585 on 4/2/19.
//

#include "GPUBasedTotalGeneralizedVariation.hpp"
#include <filesystem>

GPUBasedTGV::GPUBasedTGV(size_t argc, char **argv) {
    device = gpu::chooseGPUDevice(argc, argv);
    context.init(device.device_id_opencl);
    context.activate();
    memsize = device.mem_size;
}

void GPUBasedTGV::initKernels() {
    tgvEpsilonKernel = ocl::Kernel(epsilon_kernel, epsilon_kernel_length, "epsilon");
    tgvGradientKernel = ocl::Kernel(gradient_kernel, gradient_kernel_length, "gradient");
    tgvCopyKernel = ocl::Kernel(copy_kernel, copy_kernel_length, "copy");
    tgvMulMatrixOnConstantKernel = ocl::Kernel(mul_matrix_on_constant_kernel, mul_matrix_on_constant_kernel_length,
                                               "mulMatrixOnConstant");
    tgvProjectKernel = ocl::Kernel(project_kernel, project_kernel_length, "project");
    tgvSumOfMatrixKernel = ocl::Kernel(sum_of_matrix_kernel, sum_of_matrix_kernel_length, "sumOfMatrix");
    tgvTranspondedGradientKernel = ocl::Kernel(transponded_gradient_kernel, transponded_gradient_kernel_length,
                                               "transpondedGradient");
    tgvTranspondedEpsilonKernel = ocl::Kernel(transponded_epsilon_kernel, transponded_epsilon_kernel_length,
                                              "transpondedEpsilon");
    tgvCalculateHistKernel = ocl::Kernel(calculate_hist_kernel, calculate_hist_kernel_length, "calculateHist");
    tgvProxKernel = ocl::Kernel(prox_kernel, prox_kernel_length, "prox");
    tgvEpsilonKernel.compile();
    tgvGradientKernel.compile();
    tgvCopyKernel.compile();
    tgvMulMatrixOnConstantKernel.compile();
    tgvProjectKernel.compile();
    tgvSumOfMatrixKernel.compile();
    tgvTranspondedGradientKernel.compile();
    tgvTranspondedEpsilonKernel.compile();
    tgvCalculateHistKernel.compile();
    tgvProxKernel.compile();
}

void GPUBasedTGV::init() {
    initKernels();
    auto resultOfLoading = loadImages("data");
    loadData(observations, std::get<5>(resultOfLoading));
    loadData(image, std::get<4>(resultOfLoading));
    width = std::get<2>(resultOfLoading);
    height = std::get<3>(resultOfLoading);
    amountOfObservation = std::get<1>(resultOfLoading) - 1;
    size_t sizeOfImage = height * width;
    reserveDataN(imageDual, sizeOfImage);
    reserveDataN(v, sizeOfImage * 2);
    reserveDataN(vDual, sizeOfImage * 2);
    reserveDataN(p, sizeOfImage * 2);
    reserveDataN(pDual, sizeOfImage * 2);
    reserveDataN(q, sizeOfImage * 4);
    reserveDataN(qDual, sizeOfImage * 4);
    reserveDataN(transpondedGradient, sizeOfImage);
    reserveDataN(histogram, amountOfObservation * sizeOfImage);
    reserveDataN(prox, 2 * amountOfObservation * sizeOfImage);
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

void GPUBasedTGV::start(size_t iterations, float tau, float lambda_tv, float lambda_tgv, float lambda_data) {
    std::vector<float> result = getBuffer(image);
    unsigned int workGroupSize = 128;
    std::cout << "Image size: " << memoryBuffers[image].first << std::endl;
    unsigned int globalWorkSize = (memoryBuffers[image].first + workGroupSize - 1) / workGroupSize * workGroupSize;
    std::cout << "Global work size: " << globalWorkSize << std::endl;
    std::cout << "Init of hist\n" << std::endl;
    ///INITIAL VALUES


    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           memoryBuffers[image].second,
                           memoryBuffers[v].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[image].first);
    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           memoryBuffers[image].second,
                           memoryBuffers[p].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[image].first);
    tgvEpsilonKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                          memoryBuffers[p].second,
                          memoryBuffers[q].second, (unsigned int) width, (unsigned int) height,
                          (unsigned int) memoryBuffers[image].first);
    ///
    for (size_t i = 0; i < iterations; i++) {
        std::cout << "Iteration #: " << i << std::endl;
        iteration(tau, lambda_tv, lambda_tgv, lambda_data, workGroupSize, globalWorkSize);
    }
    result = getBuffer(image);
}

void GPUBasedTGV::writeResult(const std::string &name) {
    auto result = getBuffer(image);
    unsigned char *image = new unsigned char[result.size()];
    for (size_t i = 0; i < result.size(); i++) {
        image[i] = (unsigned char) result[i];
    }
    stbi_write_png(name.c_str(), width, height, 1, image, width);
    delete[](image);
}

std::vector<float> GPUBasedTGV::getImage() {
    return getBuffer(image);
}

void GPUBasedTGV::iteration(float tau, float lambda_tv, float lambda_tgv, float lambda_data, unsigned int workGroupSize,
                            unsigned int globalWorkSize) {
    float tau_u, tau_v, tau_p, tau_q;
    tau_u = tau;
    tau_v = tau;
    tau_p = tau;
    tau_q = tau;

    ///UN
    tgvTranspondedGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[p].second,
                                      memoryBuffers[transpondedGradient].second, (unsigned int) width,
                                      (unsigned int) height,
                                      (unsigned int) memoryBuffers[image].first);
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[transpondedGradient].second, (-tau_u * lambda_tv),
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[transpondedGradient].second,
                              memoryBuffers[image].second, (unsigned int) width, (unsigned int) height, 1,
                              (unsigned int) memoryBuffers[image].first);
    tgvProxKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[observations].second,
                       memoryBuffers[histogram].second, memoryBuffers[imageDual].second, memoryBuffers[prox].second,
                       (unsigned int) width, (unsigned int) height,
                       (unsigned int) memoryBuffers[image].first, (unsigned int) amountOfObservation, tau, lambda_data);
    auto temp = getBuffer(imageDual);
    /// VN
    tgvTranspondedEpsilonKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[q].second,
                                     memoryBuffers[transpondedEpsilon].second, (unsigned int) width,
                                     (unsigned int) height,
                                     (unsigned int) memoryBuffers[image].first);
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[transpondedEpsilon].second, (-tau_v * lambda_tgv),
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[v].second,
                       memoryBuffers[vDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[vDual].second,
                              memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[p].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[pDual].second, lambda_tv,
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[vDual].second,
                              memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    ///

    ///PN
    //Копируем imageDual в transpondedGradient
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[imageDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[transpondedGradient].second, (unsigned int) width, (unsigned int) height, 1,
                       (unsigned int) memoryBuffers[image].first);
    //Умножаем на -2
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[transpondedGradient].second, -2,
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    //Добавляем Image
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[transpondedGradient].second,
                              memoryBuffers[image].second, (unsigned int) width, (unsigned int) height, 1,
                              (unsigned int) memoryBuffers[image].first);
    //Считаем градиент пишем в transpondedEpsilon
    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           memoryBuffers[transpondedGradient].second,
                           memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[image].first);
    //Умножаем на -1
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[transpondedEpsilon].second, -1,
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    //Копируем vn в pn
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[vDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    //Умножаем на -2
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[pDual].second, -1,
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    //сложим Gradient((-2 * un) + u) +  (-2 * vn)
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[pDual].second,
                              memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //добавим v
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[pDual].second,
                              memoryBuffers[v].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //умножим на tau_p *lambda_tv
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[pDual].second, tau_p * lambda_tv,
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[pDual].second,
                              memoryBuffers[p].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);

    tgvProjectKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[pDual].second,
                          (unsigned int) width, (unsigned int) height, (unsigned int) memoryBuffers[image].first, 2,
                          lambda_tv);

    ///

    ///QN
    //Копируем vDual в transpondedEpsilon
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[vDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    //Добавим v
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[transpondedEpsilon].second,
                              memoryBuffers[v].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //Посчитаем эпсилон в qn
    tgvEpsilonKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                          memoryBuffers[transpondedEpsilon].second,
                          memoryBuffers[qDual].second, (unsigned int) width, (unsigned int) height,
                          (unsigned int) memoryBuffers[image].first);
    //Умножим на (-tau_q * lambda_tgv)
    tgvMulMatrixOnConstantKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                      memoryBuffers[qDual].second, -tau_q * lambda_tgv,
                                      (unsigned int) width, (unsigned int) height, 4,
                                      (unsigned int) memoryBuffers[image].first);
    //добавим q
    tgvSumOfMatrixKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[qDual].second,
                              memoryBuffers[q].second, (unsigned int) width, (unsigned int) height, 4,
                              (unsigned int) memoryBuffers[image].first);
    //project
    tgvProjectKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[qDual].second,
                          (unsigned int) width, (unsigned int) height, (unsigned int) memoryBuffers[image].first, 4,
                          lambda_tgv);
    ///

    /// Move to previus location
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[image].second,
                       memoryBuffers[imageDual].second, (unsigned int) width, (unsigned int) height, 1,
                       (unsigned int) memoryBuffers[image].first);
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[v].second,
                       memoryBuffers[vDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[p].second,
                       memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    tgvCopyKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                       memoryBuffers[q].second,
                       memoryBuffers[qDual].second, (unsigned int) width, (unsigned int) height, 4,
                       (unsigned int) memoryBuffers[image].first);
    ///
}
//Gradient pn = project(p + (tau_p *lambda_tv) * (-mathRoutine::calculateGradient((-2 * un) + u) +  (-2 * vn) +  v),lambda_tv); //возможно нельзя вытаскивать знак из-под градиента
//Epsilon qn = project(q + (-tau_q * lambda_tgv) * mathRoutine::calculateEpsilon(-2 * vn + v), lambda_tgv);// один буффер размера v и один размера u, calculateGradient можно пока пихнуть в transponded epsilon