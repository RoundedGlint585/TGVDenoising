//
// Created by roundedglint585 on 4/2/19.
//

#include "GPUBasedTotalGeneralizedVariation.hpp"


GPUBasedTGV::GPUBasedTGV(std::size_t index) : workGroupSize(128) {
    char **arg = (char **) calloc(2, sizeof(char *));
    arg[1] = (char *) (to_string(index).c_str());
    device = gpu::chooseGPUDevice(2, arg);
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
    tgvAnormKernel.compile();
}


void GPUBasedTGV::init(const std::vector<float> &observations, size_t height, size_t width) {
    initKernels();
    this->width = width;
    this->height = height;
    size_t imageSize = width * height;
    this->amountOfObservation = observations.size() / imageSize;
    std::cout << amountOfObservation << std::endl;
    std::vector<float> image(imageSize, 0.0f);
    for (size_t i = 0; i < imageSize; i++) {
        image[i] = observations[i];
    }
    loadData(this->observations, observations);
    loadData(this->image, image);
    reserveDataN(imageDual, imageSize);
    reserveDataN(v, imageSize * 2);
    reserveDataN(vDual, imageSize * 2);
    reserveDataN(p, imageSize * 2);
    reserveDataN(pDual, imageSize * 2);
    reserveDataN(q, imageSize * 4);
    reserveDataN(qDual, imageSize * 4);
    reserveDataN(transpondedGradient, imageSize);
    reserveDataN(transpondedEpsilon, 2 * imageSize);
    reserveDataN(histogram, amountOfObservation * imageSize);
    reserveDataN(prox, 2 * amountOfObservation * imageSize);
    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           memoryBuffers[this->image].second,
                           memoryBuffers[v].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[this->image].first);
    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           memoryBuffers[this->image].second,
                           memoryBuffers[p].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[this->image].first);
    tgvEpsilonKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                          memoryBuffers[p].second,
                          memoryBuffers[q].second, (unsigned int) width, (unsigned int) height,
                          (unsigned int) memoryBuffers[this->image].first);
    tgvCalculateHistKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[this->observations].second,
                                memoryBuffers[histogram].second, (unsigned int) width, (unsigned int) height,
                                (unsigned int) memoryBuffers[this->image].first, (unsigned int) amountOfObservation);
}


void GPUBasedTGV::init(size_t amountOfImages, size_t width, size_t height, const std::vector<float> &image,
                       const std::vector<float> &observations) {
    initKernels();
    loadData(this->observations, observations);
    loadData(this->image, image);
    this->width = width;
    this->height = height;
    amountOfObservation = amountOfImages;
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
    workGroupSize = 128;
    globalWorkSize = (memoryBuffers[this->image].first + workGroupSize - 1) / workGroupSize * workGroupSize;
    ///INITIAL VALUES
    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           memoryBuffers[this->image].second,
                           memoryBuffers[v].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[this->image].first);
    tgvGradientKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           memoryBuffers[this->image].second,
                           memoryBuffers[p].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[this->image].first);
    tgvEpsilonKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                          memoryBuffers[p].second,
                          memoryBuffers[q].second, (unsigned int) width, (unsigned int) height,
                          (unsigned int) memoryBuffers[this->image].first);
    tgvCalculateHistKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), memoryBuffers[this->observations].second,
                                memoryBuffers[histogram].second, (unsigned int) width, (unsigned int) height,
                                (unsigned int) memoryBuffers[this->image].first, (unsigned int) amountOfObservation);
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


std::vector<float> GPUBasedTGV::getImage() const {
    return getBuffer(image);
}

size_t GPUBasedTGV::getHeight() const {
    return height;
}

size_t GPUBasedTGV::getWidth() const {
    return width;
}

void
GPUBasedTGV::calculateImageDual(float tau_u, float lambda_tv, float tau, float lambda_data) {
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

void GPUBasedTGV::calculateVDual(float tau_v, float lambda_tgv, float lambda_tv) {
    //v + (-lambda_tgv *tau_v) * mathRoutine::calculateTranspondedEpsilon(q) + (tau_v*lambda_tv )* p;
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    // mathRoutine::calculateTranspondedEpsilon(q)
    tgvTranspondedEpsilonKernel.exec(workSize, memoryBuffers[q].second,
                                     memoryBuffers[transpondedEpsilon].second, (unsigned int) width,
                                     (unsigned int) height,
                                     (unsigned int) memoryBuffers[image].first);
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

void GPUBasedTGV::calculatePDual(float tau_p, float lambda_tv) {
    //project(p + (tau_p * lambda_tv) * (mathRoutine::calculateGradient((-1) * ((-2) * un + u)) +  ((-2) * vn +  v)), lambda_tv);
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    //Копируем imageDual в transpondedGradient
    tgvCopyKernel.exec(workSize,
                       memoryBuffers[imageDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[transpondedGradient].second, (unsigned int) width, (unsigned int) height, 1,
                       (unsigned int) memoryBuffers[image].first);
    //Mul un on (-2) = (-2 * un)
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[transpondedGradient].second, -2.f,
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    //Add Image  = (-2*un)+u
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[transpondedGradient].second,
                              memoryBuffers[image].second, (unsigned int) width, (unsigned int) height, 1,
                              (unsigned int) memoryBuffers[image].first);
    //Mul on-1 =  ((-1) * ((-2) * un + u))
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[transpondedGradient].second, -1.f,
                                      (unsigned int) width, (unsigned int) height, 1,
                                      (unsigned int) memoryBuffers[image].first);
    //calculate gradient and write down into transpondedEpsilon = mathRoutine::calculateGradient((-1) * ((-2) * un + u))
    tgvGradientKernel.exec(workSize,
                           memoryBuffers[transpondedGradient].second,
                           memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height,
                           (unsigned int) memoryBuffers[image].first);
    //Copy vn into pn
    tgvCopyKernel.exec(workSize,
                       memoryBuffers[vDual].second, //temporary copy p to multiply on lambda_tv
                       memoryBuffers[pDual].second, (unsigned int) width, (unsigned int) height, 2,
                       (unsigned int) memoryBuffers[image].first);
    //Mul on -2  = ((-2) * vn)
    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[pDual].second, -2.f,
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    //Add v =  ((-2) * vn + v))
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[pDual].second,
                              memoryBuffers[v].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //add (mathRoutine::calculateGradient((-1) * ((-2) * un + u)) to  ((-2) * vn +  v))
    tgvSumOfMatrixKernel.exec(workSize, memoryBuffers[pDual].second,
                              memoryBuffers[transpondedEpsilon].second, (unsigned int) width, (unsigned int) height, 2,
                              (unsigned int) memoryBuffers[image].first);
    //Mul on  tau_p *lambda_tv = (tau_p * (lambda_tv * ((-1)* mathRoutine::calculateGradient(2 * un +  u) + ((-2) * vn + v))))

    tgvMulMatrixOnConstantKernel.exec(workSize,
                                      memoryBuffers[pDual].second, tau_p * lambda_tv,
                                      (unsigned int) width, (unsigned int) height, 2,
                                      (unsigned int) memoryBuffers[image].first);
    //Add p
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

void GPUBasedTGV::calculateQDual(float tau_q, float lambda_tgv) {
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

void GPUBasedTGV::iteration(float tau, float lambda_tv, float lambda_tgv, float lambda_data) {

    float tau_u, tau_v, tau_p, tau_q;
    tau_u = tau;
    tau_v = tau;
    tau_p = tau;
    tau_q = tau;
    auto workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    ///UN
    calculateImageDual(tau_u, lambda_tv, tau, lambda_data);
    ///

    /// VN
    calculateVDual(tau_v, lambda_tgv, lambda_tv);
    ///

    ///PN
    calculatePDual(tau_p, lambda_tv);
    ///

    ///QN
    calculateQDual(tau_q, lambda_tgv);
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