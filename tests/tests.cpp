//
// Created by roundedglint585 on 3/19/19.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <filesystem>
#include <gtest/gtest.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "../src/commonKernels.hpp"
#include "../src/MathRoutine.hpp"

class imageTest : public ::testing::Test {
protected:
    void SetUp() override {

    }

    void TearDown() override {

    }

    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    mathRoutine::Gradient correctGradient = {{{0.f,   0},     {0,     15.f},  {0,      0},     {0,      0},      {0, 0}},
                                             {{15.f,  15.f},  {-15.f, -15.f}, {0,      0},     {0,      195.f},  {0, 0}},
                                             {{-15.f, -15.f}, {0,     0},     {195.f,  0},     {-195.f, 4.f},    {0, 0}},
                                             {{0,     0},     {0,     195.f}, {199.f,  199.f}, {-199.f, -197.f}, {0, 0}},
                                             {{195.f, 0},     {4.0f,  0},     {-197.f, 0},     {-2.f,   0},      {0, 0}}};
    mathRoutine::Epsilon correctEpsilon = {{{0,      15.f,  15.f,  15.f},  {0,      -15.f, -15.f, -30.f},  {0,      0,      0,      0},      {0,     0,      0,      195.f},  {0, 0, 0, 0}},
                                           {{-30.f,  -30.f, -30.f, -30.f}, {15.f,   15.f,  15.f,  15.f},   {0,      195.f,  195.f,  0},      {0,     -195.f, -195.f, -191.f}, {0, 0, 0, 0}},
                                           {{15.f,   15.f,  15.f,  15.f},  {195.f,  0,     0,     195.f},  {-390.f, 4.f,    4.f,    199.f},  {195.f, -4.f,   -4.f,   -201.f}, {0, 0, 0, 0}},
                                           {{0,      195.f, 195.f, 0},     {199.f,  4.f,   4.f,   -195.f}, {-398.f, -396.f, -396.f, -199.f}, {199.f, 197.f,  197.f,  197.f},  {0, 0, 0, 0}},
                                           {{-191.f, 0,     0,     0},     {-201.f, 0,     0,     0},      {195.f,  0,      0,      0},      {2.f,   0,      0,      0},      {0, 0, 0, 0},}};
    mathRoutine::Image correctTranspondedGradient = {{0,      -15.f,  0,      0,      0},
                                                     {-30.f,  60.f,   -15.f,  -195.f, 0},
                                                     {45.f,   -30.f,  -195.f, 581.f,  -195.f},
                                                     {-15.f,  -195.f, -398.f, 599.f,  -199.f},
                                                     {-195.f, 386.f,  400.f,  -392.f, -2.f}};
    mathRoutine::Gradient correctTransopondedEpsilon = {{{-15.f,  -30.f},  {15.f,   60.f},   {0,      -15.f},  {0,      -195.f}, {0,     0}},
                                                        {{75.f,   75.f},   {-75.f,  -90.f},  {-180.f, -180.f}, {195.f,  776.f},  {0,     -195.f}},
                                                        {{-60.f,  -60.f},  {-165.f, -165.f}, {776.f,  -203.f}, {-776.f, 18.f},   {195.f, -4.f}},
                                                        {{-180.f, -180.f}, {-203.f, 581.f},  {997.f,  798.f},  {-798.f, -991.f}, {199.f, 197.f}},
                                                        {{386.f,  0},      {14.f,   -195.f}, {-792.f, -199.f}, {390.f,  197.f},  {2.f,   0}}};
    mathRoutine::Image correctNormalizedGradient = {{0,          15.f,       0,         0,          0,},
                                                    {21.213203f, 21.213203f, 0,         195.f,      0},
                                                    {21.213203f, 0,          195.f,     195.04102f, 0},
                                                    {0,          195.f,      281.4285f, 280.01785f, 0},
                                                    {195.f,      4.f,        197.f,     2.f,        0,}};
    mathRoutine::Image correctNormalizedEpsilon = {{25.9808, 36.7423, 0,       195,     0},
                                                   {60,      30,      275.772, 335.456, 0},
                                                   {30,      275.772, 437.873, 280.104, 0},
                                                   {275.772, 278.672, 715.288, 395.004, 0},
                                                   {191,     201,     195,     2,       0}};
    mathRoutine::Gradient correctProjectedGradient = {{{0,         0},         {0,         2.f},       {0,        0},        {0,         0},         {0, 0},},
                                                      {{1.41421f,  1.41421f},  {-1.41421f, -1.41421f}, {0,        0},        {0,         2.f},       {0, 0},},
                                                      {{-1.41421f, -1.41421f}, {0,         0},         {2.f,      0},        {-1.99958f, 0.041017f}, {0, 0},},
                                                      {{0,         0},         {0,         2.f},       {1.41421f, 1.41421f}, {-1.42134f, -1.40705f}, {0, 0},},
                                                      {{2.f,       0},         {2.f,       0},         {-2.f,     0},        {-2.f,      0},         {0, 0},}};
};

TEST_F(imageTest, readImageFromFile) {
    int width, height, channels;
    unsigned char *image = stbi_load("../tests/test_1.png",
                                     &width,
                                     &height,
                                     &channels,
                                     STBI_grey);
    ASSERT_FALSE(image == nullptr) << "Image is not loaded";

    mathRoutine::Image imageInMatrix = mathRoutine::createImageFromUnsignedCharArray(image, width, height);
    std::size_t returnedWidth, returnedHeight;
    unsigned char *result = mathRoutine::getArrayFromImage(&returnedWidth, &returnedHeight, imageInMatrix);
    ASSERT_EQ(width, returnedWidth);
    ASSERT_EQ(height, returnedHeight);
    for (size_t i = 0; i < height * width; i++) {
        ASSERT_EQ(result[i], image[i]);
    }

    stbi_image_free(image);
    delete[] result;
}

TEST_F(imageTest, writeAndLoad) {
    int width, height, channels;
    unsigned char *image = stbi_load("../tests/test_1.png",
                                     &width,
                                     &height,
                                     &channels,
                                     STBI_grey);
    ASSERT_FALSE(image == nullptr) << "Image is not loaded";
    mathRoutine::Image imageInMatrix = mathRoutine::createImageFromUnsignedCharArray(image, width, height);
    std::size_t returnedWidth, returnedHeight;
    unsigned char *result = mathRoutine::getArrayFromImage(&returnedWidth, &returnedHeight, imageInMatrix);
    stbi_write_png("result1.png", returnedWidth, returnedHeight, STBI_grey, image, returnedWidth);
    unsigned char *imageLoadedAgain = stbi_load("result1.png",
                                                &width,
                                                &height,
                                                &channels,
                                                STBI_grey);
    ASSERT_FALSE(imageLoadedAgain == nullptr) << "Loaded again image is not loaded";
    for (size_t i = 0; i < width * height; i++) {
        ASSERT_EQ(result[i], image[i]);
    }


}

TEST_F(imageTest, calculateGradient) {

    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);

    ASSERT_TRUE(gradient == correctGradient) << "Gradient calculation error";
}

TEST_F(imageTest, calculateEpsilon) {
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    ASSERT_TRUE(epsilon == correctEpsilon) << "Epsilon calculation error";
}

TEST_F(imageTest, calculateTranspondedGradient) {

    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Image transpondedGradient = mathRoutine::calculateTranspondedGradient(gradient);
    ASSERT_TRUE(transpondedGradient == correctTranspondedGradient) << "Transpoded Gradient calculation error";
}

TEST_F(imageTest, calculateTranspondedEpsilon) {
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Gradient transpondedEpsilon = mathRoutine::calculateTranspondedEpsilon(epsilon);
    ASSERT_TRUE(transpondedEpsilon == correctTransopondedEpsilon) << "transponded epsilon calculation error";
}

TEST_F(imageTest, calculateAnorm) {

    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Image normalizedGradient = mathRoutine::anorm(gradient);
    for (size_t i = 0; i < normalizedGradient.size(); i++) {
        for (size_t j = 0; j < normalizedGradient[0].size(); j++) {
            ASSERT_NEAR(normalizedGradient[i][j], correctNormalizedGradient[i][j], mathRoutine::eps);
        }
    }
    mathRoutine::Image normalizedEpsilon = mathRoutine::anorm(epsilon);
    for (size_t i = 0; i < normalizedEpsilon.size(); i++) {
        for (size_t j = 0; j < normalizedEpsilon[0].size(); j++) {
            ASSERT_NEAR(normalizedEpsilon[i][j], correctNormalizedEpsilon[i][j], mathRoutine::eps);
        }
    }
}

TEST_F(imageTest, projectOfMatrixTest) {
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Gradient projectedGradient = mathRoutine::project(gradient, 2);

    for (size_t i = 0; i < projectedGradient.size(); i++) {
        for (size_t j = 0; j < projectedGradient[0].size(); j++) {
            ASSERT_NEAR(projectedGradient[i][j][0], correctProjectedGradient[i][j][0], mathRoutine::eps);
            ASSERT_NEAR(projectedGradient[i][j][1], correctProjectedGradient[i][j][1], mathRoutine::eps);
        }
    }
}

TEST_F(imageTest, sumOfImages) {
    using namespace mathRoutine;
    mathRoutine::Image first = {{1, 2, 4},
                                {0, 0, 0},
                                {2, 6, 1}};
    mathRoutine::Image second = {{5, -2, 1},
                                 {0, -4, 0},
                                 {0, 1,  1}};
    mathRoutine::Image result = {{6, 0,  5},
                                 {0, -4, 0},
                                 {2, 7,  2}};
    ASSERT_EQ(result, first + second) << "sum of matrix is not equal";
}

class GPUImageTest : public ::testing::Test {
protected:
    GPUImageTest() {
        char **arg = (char **) calloc(2, sizeof(char *));
        arg[1] = (char *) "0";
        device = gpu::chooseGPUDevice(2, arg);
        context.init(device.device_id_opencl);
        context.activate();
    }

    void SetUp() override {
        imageBuf.resizeN(image.size());
        imageBuf.writeN(image.data(), image.size());
    }

    void TearDown() override {

    }

    std::vector<mathRoutine::Image> getImagesFromPath(const std::string &path) {
        std::vector<mathRoutine::Image> result;

        using namespace std::filesystem;
        for (auto &p: directory_iterator(path)) {
            std::cout << "loading image: " << p << std::endl;
            std::string name = p.path();
            int width, height, channels;
            unsigned char *loadedImage = stbi_load(name.c_str(),
                                                   &width,
                                                   &height,
                                                   &channels,
                                                   STBI_grey);
            mathRoutine::Image imageRes = mathRoutine::createImageFromUnsignedCharArray(loadedImage, width, height);
            result.emplace_back(imageRes);
            stbi_image_free(loadedImage);
        }
        return result;
    }

    std::vector<float> getImagesFromPathPlain(const std::string &path) {
        using namespace std::filesystem;
        size_t totalSize = 0;
        size_t amountOfImages = 0;
        int width, height;
        int channels;
        std::vector<float> observations;
        for (auto &p: directory_iterator(path)) {
            std::cout << "loading image: " << p << std::endl;
            std::string name = p.path();
            unsigned char *bytes = stbi_load(name.c_str(),
                                             &width,
                                             &height,
                                             &channels,
                                             STBI_grey);

            for (size_t i = 0; i < width * height; i++) {
                observations.emplace_back((float) bytes[i]);
            }
        }
        return observations;
    }

    gpu::Device device;
    gpu::Context context;
    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    std::vector<float> image = {1, 1, 1, 1, 1,
                                1, 16, 1, 1, 1,
                                16, 1, 1, 196, 1,
                                1, 1, 1, 200, 1,
                                1, 196, 200, 3, 1};

    unsigned int workGroupSize = 128;
    unsigned int globalWorkSize = (image.size() + workGroupSize - 1) / workGroupSize * workGroupSize;
    gpu::WorkSize workSize = gpu::WorkSize(workGroupSize, globalWorkSize);
    ocl::Kernel tgvGradientKernel = ocl::Kernel(gradient_kernel, gradient_kernel_length, "gradient");
    ocl::Kernel tgvEpsilonKernel = ocl::Kernel(epsilon_kernel, epsilon_kernel_length, "epsilon");
    ocl::Kernel tgvTranspondedGradientKernel = ocl::Kernel(transponded_gradient_kernel,
                                                           transponded_gradient_kernel_length, "transpondedGradient");
    ocl::Kernel tgvTranspondedKernel = ocl::Kernel(transponded_epsilon_kernel, transponded_epsilon_kernel_length,
                                                   "transpondedEpsilon");
    ocl::Kernel tgvProjectedKernel = ocl::Kernel(project_kernel, project_kernel_length, "project");
    ocl::Kernel tgvCalculateHist = ocl::Kernel(calculate_hist_kernel, calculate_hist_kernel_length, "calculateHist");
    ocl::Kernel tgvAnormKernel = ocl::Kernel(anorm_kernel, anorm_kernel_length, "anorm");
    gpu::gpu_mem_32f imageBuf;

};


TEST_F(GPUImageTest, gpuProjectOfMatrixTest) {
    gpu::gpu_mem_32f gradientBuf;
    gradientBuf.resizeN(2 * image.size());
    gpu::gpu_mem_32f normedBuf;
    normedBuf.resizeN(image.size());
    tgvGradientKernel.compile();
    tgvAnormKernel.compile();
    tgvProjectedKernel.compile();
    tgvGradientKernel.exec(workSize, imageBuf, gradientBuf, (int) imageInMatrix.size(), (int) imageInMatrix[0].size(),
                           (int) image.size());

    tgvAnormKernel.exec(workSize, gradientBuf, normedBuf, (int) imageInMatrix.size(), (int) imageInMatrix[0].size(),
                        (int) image.size(), 2, 1.f);
    std::vector<float> resultNormed(image.size(), 0.f);
    normedBuf.readN(resultNormed.data(), resultNormed.size());
    mathRoutine::Gradient correctGradient = mathRoutine::calculateGradient(imageInMatrix);
    auto correctNormed = mathRoutine::anorm(correctGradient);
    auto projected = mathRoutine::project(correctGradient, 1.f);
    tgvProjectedKernel.exec(workSize, gradientBuf, normedBuf, (int) imageInMatrix.size(), (int) imageInMatrix[0].size(),
                            (int) image.size(), 2);
    std::vector<float> result(2 * image.size(), 0.f);
    gradientBuf.readN(result.data(), result.size());

    for (size_t i = 0; i < projected.size(); i++) {
        for (size_t j = 0; j < projected[i].size(); j++) {
            for (size_t k = 0; k < projected[i][j].size(); k++) {
                float first = projected[i][j][k];
                float second = result[j + i * projected[i].size() + k * image.size()];
                ASSERT_NEAR(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second),
                            9); // OpenCL provide precision to 3 ulp by IEEE standart for operations like sqrt,/ and etc. in this case there are 2 of these operations
            }
        }
    }
}


TEST_F(GPUImageTest, gpuCalculateGradient) {
    gpu::gpu_mem_32f gradientBuf;
    gradientBuf.resizeN(2 * image.size());
    tgvGradientKernel.compile();
    tgvGradientKernel.exec(workSize,
                           imageBuf,
                           gradientBuf, (int) imageInMatrix.size(), (int) imageInMatrix[0].size(),
                           (int) image.size());
    std::vector<float> result(2 * image.size(), 0.f);
    gradientBuf.readN(result.data(), 2 * image.size());
    mathRoutine::Gradient correctGradient = mathRoutine::calculateGradient(imageInMatrix);
    for (size_t i = 0; i < correctGradient.size(); i++) {
        for (size_t j = 0; j < correctGradient[0].size(); j++) {
            float first = result[j + i * correctGradient[0].size()];
            float second = correctGradient[i][j][0];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second)); // Should be equals to 0 ulps
            first = result[j + i * correctGradient[0].size() + result.size() / 2];
            second =  correctGradient[i][j][1];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second));
        }
    }
}

TEST_F(GPUImageTest, gpuCalculateEpsilon) {
    gpu::gpu_mem_32f gradientBuf;
    gradientBuf.resizeN(2 * image.size());
    gpu::gpu_mem_32f epsilonBuf;
    epsilonBuf.resizeN(4 * image.size());
    tgvGradientKernel.compile();
    tgvEpsilonKernel.compile();
    tgvGradientKernel.exec(workSize, imageBuf, gradientBuf, (int) imageInMatrix.size(), (int) imageInMatrix[0].size(),
                           (int) image.size());
    tgvEpsilonKernel.exec(workSize, gradientBuf, epsilonBuf, (unsigned int) imageInMatrix.size(),
                          (int) imageInMatrix[0].size(),
                          (int) image.size());
    std::vector<float> result(4 * image.size(), 0.f);
    epsilonBuf.readN(result.data(), 4 * image.size());
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon correctEpsilon = mathRoutine::calculateEpsilon(gradient);
    for (size_t i = 0; i < correctEpsilon.size(); i++) {
        for (size_t j = 0; j < correctEpsilon[0].size(); j++) {
            float first = result[j + i * correctEpsilon[0].size()];
            float second = correctEpsilon[i][j][0];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second)); // Should be equals to 0 ulps
            first = result[j + i * correctEpsilon[0].size() + result.size() / 4];
            second =  correctEpsilon[i][j][1];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second));
            first = result[j + i * correctEpsilon[0].size() + 2 * result.size() / 4];
            second =  correctEpsilon[i][j][2];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second));
            first = result[j + i * correctEpsilon[0].size() + result.size() / 4 * 3];
            second =  correctEpsilon[i][j][3];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second));
        }
    }
}


TEST_F(GPUImageTest, gpuCalculateTranspondedGradient) {
    gpu::gpu_mem_32f gradientBuf;
    gradientBuf.resizeN(2 * image.size());
    tgvGradientKernel.compile();
    tgvTranspondedGradientKernel.compile();
    tgvGradientKernel.exec(workSize, imageBuf, gradientBuf, (unsigned int) imageInMatrix.size(),
                           (unsigned int) imageInMatrix[0].size(),
                           (unsigned int) image.size());

    tgvTranspondedGradientKernel.exec(workSize, gradientBuf, imageBuf,
                                      (unsigned int) imageInMatrix.size(), (unsigned int) imageInMatrix[0].size(),
                                      (unsigned int) image.size());
    std::vector<float> result(image.size(), 0.f);
    imageBuf.readN(result.data(), image.size());

    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Image transpondedGradient = mathRoutine::calculateTranspondedGradient(gradient);
    for (size_t i = 0; i < transpondedGradient.size(); i++) {
        for (size_t j = 0; j < transpondedGradient[0].size(); j++) {
            float first = result[j + i * transpondedGradient[0].size()];
            if(first == -0.0f){
                first = 0.0f;
            }
            float second = transpondedGradient[i][j];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second)); // Should be equals to 0 ulps
        }
    }
}


TEST_F(GPUImageTest, gpuCalculateTranspondedEpsilon) {
    gpu::gpu_mem_32f gradientBuf;
    gradientBuf.resizeN(2 * image.size());
    gpu::gpu_mem_32f epsilonBuf;
    epsilonBuf.resizeN(4 * image.size());
    tgvGradientKernel.compile();
    tgvEpsilonKernel.compile();
    tgvTranspondedKernel.compile();
    tgvGradientKernel.exec(workSize, imageBuf, gradientBuf, (int) imageInMatrix.size(), (int) imageInMatrix[0].size(),
                           (int) image.size());
    tgvEpsilonKernel.exec(workSize, gradientBuf,
                          epsilonBuf, (int) imageInMatrix.size(), (int) imageInMatrix[0].size(),
                          (int) image.size());
    tgvTranspondedKernel.exec(workSize, epsilonBuf, gradientBuf, (unsigned int) imageInMatrix.size(),
                              (unsigned int) imageInMatrix[0].size(),
                              (unsigned int) image.size());
    std::vector<float> result(2 * image.size(), 0.f);
    gradientBuf.readN(result.data(), result.size());
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Gradient correctTransopondedEpsilon = mathRoutine::calculateTranspondedEpsilon(epsilon);
    for (size_t i = 0; i < correctTransopondedEpsilon.size(); i++) {
        for (size_t j = 0; j < correctTransopondedEpsilon[0].size(); j++) {
            float first = result[j + i * correctTransopondedEpsilon[0].size()];
            float second = correctTransopondedEpsilon[i][j][0];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second)); // Should be equals to 0 ulps
            first = result[j + i * correctTransopondedEpsilon[0].size() + image.size()];
            second = correctTransopondedEpsilon[i][j][1];
            ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second)); // Should be equals to 0 ulps
        }
    }
}


TEST_F(GPUImageTest, gpuHistsTest) {
    using namespace mathRoutine;
    std::vector<Image> images = getImagesFromPath("../tests/data");

    //histogram fot images
    size_t height = images[0].size();
    size_t width = images[0][0].size();
    std::vector<Image> Ws(images.size(), mathRoutine::Image(height, std::vector<float>(width, 0)));
    for (size_t histNum = 0; histNum < Ws.size(); histNum++) {
        for (auto &image: images) {
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    if (images[histNum][i][j] > image[i][j]) {
                        Ws[histNum][i][j]++;
                    } else {
                        Ws[histNum][i][j]--;
                    }
                }
            }
        }
    }

    std::vector<float> plainImages = getImagesFromPathPlain("../tests/data");
    tgvCalculateHist.compile();
    gpu::gpu_mem_32f observations;
    observations.resizeN(plainImages.size());
    observations.writeN(plainImages.data(), plainImages.size());
    gpu::gpu_mem_32f histogram;
    histogram.resizeN(plainImages.size());
    tgvCalculateHist.exec(gpu::WorkSize(workGroupSize, globalWorkSize), observations, histogram, (unsigned int) width,
                          (unsigned int) height,
                          (unsigned int) image.size(), (unsigned int) images.size());
    std::vector<float> result(plainImages.size(), 0.f);
    histogram.readN(result.data(), result.size());
    for (size_t k = 0; k < Ws.size(); k++) {
        for (size_t i = 0; i < Ws[k].size(); i++) {
            for (size_t j = 0; j < Ws[k][i].size(); j++) {
                float first = result[j + i * Ws[k][i].size() + k * image.size()];
                float second = Ws[k][i][j];
                ASSERT_EQ(*reinterpret_cast<uint32_t * >(&first), *reinterpret_cast<uint32_t * >(&second)); // Should be equals to 0 ulps
            }
        }
    }
}


