//
// Created by roundedglint585 on 4/2/19.
//

#include "GPUBasedTotalGeneralizedVariation.hpp"
#include <filesystem>
#include <fstream>

#define Debug

GPUBasedTGV::GPUBasedTGV(size_t argc, char **argv, size_t amountOfImagesGPU) : amountOfImagesToGPU(amountOfImagesGPU) {
    device = gpu::chooseGPUDevice(argc, argv);
    context.init(device.device_id_opencl);
    context.activate();
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
    tgvClearKernel = ocl::Kernel(clear_kernel, clear_kernel_length, "clear");
    tgvAnormKernel = ocl::Kernel(anorm_kernel, anorm_kernel_length, "anorm");
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
    tgvClearKernel.compile();
    tgvAnormKernel.compile();
}

void GPUBasedTGV::init(const std::string &path, size_t amountOfImages) {
    initKernels();
    auto resultOfLoading = loadImages(path.c_str(), amountOfImages);
    loadData(observations, std::get<5>(resultOfLoading));
    loadData(image, std::get<4>(resultOfLoading));
    width = std::get<2>(resultOfLoading);
    height = std::get<3>(resultOfLoading);
    amountOfObservation = std::get<1>(resultOfLoading);
    size_t sizeOfImage = height * width;
    reserveDataN(imageDual, sizeOfImage);
    reserveDataN(v, sizeOfImage * 2);
    reserveDataN(vDual, sizeOfImage * 2);
    reserveDataN(p, sizeOfImage * 2);
    reserveDataN(pDual, sizeOfImage * 2);
    reserveDataN(q, sizeOfImage * 4);
    reserveDataN(qDual, sizeOfImage * 4);
    reserveDataN(transpondedGradient, sizeOfImage);
    reserveDataN(transpondedEpsilon, 2 * sizeOfImage);
    reserveDataN(histogram, amountOfObservation * sizeOfImage);
    reserveDataN(prox, 2 * amountOfObservation * sizeOfImage);
    std::cout << "Memory allocated and reserved" << std::endl;
}

std::tuple<size_t, size_t, int, int, std::vector<float>, std::vector<float>>
GPUBasedTGV::loadImages(std::string_view path, size_t amountOfImages) {
    using namespace std::filesystem;
    size_t totalSize = 0;
    size_t totalAmountOfImages = 0;
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
        if (totalAmountOfImages != 0) {
            for (size_t i = 0; i < width * height; i++) {
                observations.emplace_back((float) bytes[i]);
            }
        } else {
            for (size_t i = 0; i < width * height; i++) {
                image.emplace_back((float) bytes[i]);
                observations.emplace_back((float) bytes[i]);
            }
        }
        totalAmountOfImages++;
        totalSize += width * height;
    }
    if (totalAmountOfImages <= amountOfImages) {
        return std::make_tuple(totalSize, totalAmountOfImages, width, height, image, observations);
    } else {
        std::vector<float> selectedObservation = std::vector(width * height * amountOfImages, 0.f);
        //SELECT IMAGES
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                std::vector<float> allForPixel;
                for (size_t k = 0; k < totalAmountOfImages; k++) {
                    allForPixel.emplace_back(observations[j + i * width + k * width * height]);
                }
                std::sort(allForPixel.begin(), allForPixel.end());
                size_t left = (totalAmountOfImages - amountOfImages) / 2;
                for (size_t k = 0; k < amountOfImages; k++) {
                    selectedObservation[j + i * width + k * width * height] = allForPixel[k + left];
                }
            }
        }
        return std::make_tuple(amountOfImages * height * width, amountOfImages, width, height, image,
                               selectedObservation);
    }

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
    tgvCalculateHistKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[observations].second,
                                memoryBuffers[histogram].second, (unsigned int) width, (unsigned int) height,
                                (unsigned int) memoryBuffers[image].first, (unsigned int) amountOfObservation);
    auto hist = getBuffer(histogram);
    ///
    std::ofstream out("logGPU.txt");
    for (size_t i = 0; i < iterations; i++) {
        if (i % 100 == 0) {
            std::cout << "Iteration #: " << i << std::endl;
        }
        iteration(tau, lambda_tv, lambda_tgv, lambda_data, workGroupSize, globalWorkSize);
    }
    result = getBuffer(image);
}

void GPUBasedTGV::writeImage(const std::string &name) {
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

void
GPUBasedTGV::calculateImageDual(float tau_u, float lambda_tv, float tau, float lambda_data, unsigned int workGroupSize,
                                unsigned int globalWorkSize) {
    ///prox(u + (-tau_u * lambda_tv) * (mathRoutine::calculateTranspondedGradient(p)), tau_u, lambda_data)
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    ///UN

    tgvTranspondedGradientKernel.exec(workSize, memoryBuffers[p].second,
                                      memoryBuffers[transpondedGradient].second, (unsigned int) width,
                                      (unsigned int) height,
                                      (unsigned int) memoryBuffers[image].first);

    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[transpondedGradient].second, (-tau_u * lambda_tv),
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);

    tgvSumOfMatrixKernel.exec(workSize,
                              memoryBuffers[transpondedGradient].second, //here are possible problems with accur
                              memoryBuffers[image].second, (unsigned int) width, (unsigned int) height, 1,
                              (unsigned int) memoryBuffers[image].first);
    tgvProxKernel.exec(workSize, memoryBuffers[observations].second,
                       memoryBuffers[histogram].second, memoryBuffers[transpondedGradient].second,
                       memoryBuffers[prox].second,
                       (unsigned int) width, (unsigned int) height,
                       (unsigned int) memoryBuffers[image].first, (unsigned int) amountOfObservation, tau_u,
                       lambda_data);
    tgvCopyKernel.exec(workSize, memoryBuffers[transpondedGradient].second, memoryBuffers[imageDual].second,
                       (unsigned int) width, (unsigned int) height, 1,
                       (unsigned int) memoryBuffers[image].first);
}

void GPUBasedTGV::calculateVDual(float tau_v, float lambda_tgv, float lambda_tv, unsigned int workGroupSize,
                                 unsigned int globalWorkSize) {
    //v + (-lambda_tgv *tau_v) * mathRoutine::calculateTranspondedEpsilon(q) + (tau_v*lambda_tv )* p;
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    // mathRoutine::calculateTranspondedEpsilon(q)
    tgvTranspondedEpsilonKernel.exec(workSize, memoryBuffers[q].second,
                                     memoryBuffers[transpondedEpsilon].second, (unsigned int) width,
                                     (unsigned int) height,
                                     (unsigned int) memoryBuffers[image].first);
//    std::ofstream out("logGPU.txt", std::ios_base::app);
//    out << "q" << std::endl;
//    auto im = getBuffer(q);
//    for(auto& i: im){
//        float first = i;
//        out << *reinterpret_cast<uint32_t * >(&first) << std::endl;
//    }
//    out << "TranspondedEpsilon(q)" << std::endl;
//    im = getBuffer(transpondedEpsilon);
//    for(auto& i: im){
//        float first = i;
//        out << *reinterpret_cast<uint32_t * >(&first) << std::endl;
//    }
//    //(-lambda_tgv *tau_v) * mathRoutine::calculateTranspondedEpsilon(q)
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[transpondedEpsilon].second, (-tau_v * lambda_tgv),
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    // v-> vdual
    tgvCopyKernel.exec(workSize, memoryBuffers[v].second,
                       memoryBuffers[vDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    //v + (-lambda_tgv *tau_v) * mathRoutine::calculateTranspondedEpsilon(q)
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[vDual].second,
                              memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    tgvCopyKernel.exec(workSize,
                       memoryBuffers[p].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    //(tau_v*lambda_tv )* p
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[pDual].second, (tau_v * lambda_tv),
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    //v + (-lambda_tgv *tau_v) * mathRoutine::calculateTranspondedEpsilon(q) + (tau_v*lambda_tv )* p

    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[vDual].second,
                              memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
}

void GPUBasedTGV::calculatePDual(float tau_p, float lambda_tv, unsigned int workGroupSize,
                                 unsigned int globalWorkSize) {
    //p + (tau_p * lambda_tv) * ( mathRoutine::calculateGradient(2 * un + (-1)* u) + (-1)* (2 * vn +(-1)* v))
    //project(p + (tau_p * lambda_tv) * (mathRoutine::calculateGradient((-1) * ((-2) * un + u)) +  ((-2) * vn +  v)), lambda_tv);
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    //Копируем imageDual в transpondedGradient
    tgvCopyKernel.exec(workSize,
                       memoryBuffers[imageDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[transpondedGradient].second, (unsigned int) width, (unsigned int) height, 1,
                       (unsigned int) memoryBuffers[image].first);
    //Умножаем на -2 (-2 * un)
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[transpondedGradient].second, -2.f,
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    //Добавляем Image (-2*un)+u
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[transpondedGradient].second,
                              memoryBuffers[image].second, (unsigned int) width, (unsigned int) height, 1,
                              (unsigned int) memoryBuffers[image].first);
    //Умножаем на -1 ((-1) * ((-2) * un + u))
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[transpondedGradient].second, -1.f,
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    //Считаем градиент пишем в transpondedEpsilon mathRoutine::calculateGradient((-1) * ((-2) * un + u))
    tgvGradientKernel.exec(workSize,
                           memoryBuffers[transpondedGradient].second,
                           memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[image].first);
    //Копируем vn в pn
    tgvCopyKernel.exec(workSize,
                       memoryBuffers[vDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    //Умножаем на -2 ((-2) * vn)
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[pDual].second, -2.f,
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    //добавим v ((-2) * vn + v))
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[pDual].second,
                              memoryBuffers[v].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //сложим (mathRoutine::calculateGradient((-1) * ((-2) * un + u)) +  ((-2) * vn +  v))
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[pDual].second,
                              memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //умножим на tau_p *lambda_tv(tau_p * (lambda_tv * ((-1)* mathRoutine::calculateGradient(2 * un +  u) + ((-2) * vn + v))))

    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[pDual].second, tau_p * lambda_tv,
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    //добавим p
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[pDual].second,
                              memoryBuffers[p].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    gpu::gpu_mem_32f anormed;
    anormed.resizeN(height * width);
    tgvAnormKernel.exec(workSize, memoryBuffers[pDual].second, anormed,
                        (unsigned int) width, (unsigned int) height, (unsigned int) memoryBuffers[image].first, 2,
                        (float) lambda_tv);
    tgvProjectKernel.exec(workSize, memoryBuffers[pDual].second, anormed,
                          (unsigned int) width, (unsigned int) height, (unsigned int) memoryBuffers[image].first, 2U);

}

void GPUBasedTGV::calculateQDual(float tau_q, float lambda_tgv, unsigned int workGroupSize,
                                 unsigned int globalWorkSize) {
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    ///project(q + (-tau_q * lambda_tgv) * mathRoutine::calculateEpsilon(-2 * vn + v), lambda_tgv);
    //Копируем vDual в transpondedEpsilon
    tgvCopyKernel.exec(workSize,
                       memoryBuffers[vDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[transpondedEpsilon].second, -2.f,
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    //Добавим v
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[transpondedEpsilon].second,
                              memoryBuffers[v].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //Посчитаем эпсилон в qn
    tgvEpsilonKernel.exec(workSize,
                          memoryBuffers[transpondedEpsilon].second,
                          memoryBuffers[qDual].second, (unsigned int) width, (unsigned int) height,
                          (unsigned int) memoryBuffers[image].first);
    //Умножим на (-tau_q * lambda_tgv)
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[qDual].second, -tau_q * lambda_tgv,
                                      (unsigned int) width, (unsigned int) height, 4,
                                      (unsigned int) memoryBuffers[image].first);
    //добавим q
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[qDual].second,
                              memoryBuffers[q].second, (unsigned int) width, (unsigned int) height, 4,
                              (unsigned int) memoryBuffers[image].first);
    gpu::gpu_mem_32f anormed;
    anormed.resizeN(height * width);
    tgvAnormKernel.exec(workSize, memoryBuffers[qDual].second, anormed,
                        (unsigned int) width, (unsigned int) height, (unsigned int) memoryBuffers[image].first, 4, 1.f);

    //project
    tgvProjectKernel.exec(workSize, memoryBuffers[qDual].second, anormed,
                          (unsigned int) width, (unsigned int) height, (unsigned int) memoryBuffers[image].first, 4);
}

void GPUBasedTGV::iteration(float tau, float lambda_tv, float lambda_tgv, float lambda_data, unsigned int workGroupSize,
                            unsigned int globalWorkSize) {

    float tau_u, tau_v, tau_p, tau_q;
    tau_u = tau;
    tau_v = tau;
    tau_p = tau;
    tau_q = tau;
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    ///UN
    calculateImageDual(tau_u, lambda_tv, tau, lambda_data, workGroupSize, globalWorkSize);
    ///

    /// VN
    calculateVDual(tau_v, lambda_tgv, lambda_tv, workGroupSize, globalWorkSize);
    ///

    ///PN
    calculatePDual(tau_p, lambda_tv, workGroupSize, globalWorkSize);
    ///

    ///QN
    calculateQDual(tau_q, lambda_tgv, workGroupSize, globalWorkSize);
    ///


    /// Move to prev location
    tgvCopyKernel.exec(workSize,
                       memoryBuffers[imageDual].second,
                       memoryBuffers[image].second, (unsigned int) width, (unsigned int) height, (unsigned int) 1,
                       (unsigned int) memoryBuffers[image].first);

    tgvCopyKernel.exec(workSize,
                       memoryBuffers[vDual].second,
                       memoryBuffers[v].second, (unsigned int) width, (unsigned int) height, (unsigned int) 2,
                       (unsigned int) memoryBuffers[image].first);

    tgvCopyKernel.exec(workSize,
                       memoryBuffers[pDual].second,
                       memoryBuffers[p].second, (unsigned int) width, (unsigned int) height, (unsigned int) 2,
                       (unsigned int) memoryBuffers[image].first);

    tgvCopyKernel.exec(workSize,
                       memoryBuffers[qDual].second,
                       memoryBuffers[q].second, (unsigned int) width, (unsigned int) height, (unsigned int) 4,
                       (unsigned int) memoryBuffers[image].first);
}

void GPUBasedTGV::writePly(const std::string &name) {
    std::ofstream out(name.c_str());
    out << "ply" << std::endl << "format ascii 1.0" << std::endl;
    out << "element vertex " << height*width << std::endl;
    out << "property float x" << std::endl << "property float y" << std::endl << "property float z" << std::endl;
    out << "element face " << ((height-1)*(width-1)*2) << std::endl;
    out << "property list uint8 int32 vertex_indices" << std::endl << "end_header" << std::endl;
    auto result = getBuffer(image);
    for(size_t i = 0; i < height; i++){
        for(size_t j = 0; j < width; j++){
            out << i << " " << j << " " << (size_t)result[j+i*width] << std::endl;
        }
    }
    for(size_t i = 0; i < height-1; i++){
        for(size_t j = 0; j < width-1; j++){
            auto a = j + i*width;
            auto b = j+1 +i*width;
            auto c = j+width+i*width;
            auto d  = j + width+1 + i*width;
            out << "3 " << (int) a << " " << (int) b << " " << (int) c << std::endl;
            out << "3 " << (int) c << " " << (int) b << " " << (int) d << std::endl;
        }
    }
}
