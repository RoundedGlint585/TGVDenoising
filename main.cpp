#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <filesystem>
#include <time.h>
//#define CPU


#ifdef CPU

#include "src/TotalGeneralizedVariation.hpp"

#else

#include "src/GPUBasedTotalGeneralizedVariation.hpp"

#endif


#ifdef CPU

std::vector<mathRoutine::Image> prepareImages(std::string_view path, size_t amountOfImages) {
    std::vector<mathRoutine::Image> result;
    std::vector<std::vector<std::vector<float>>> transformed;
    using namespace std::filesystem;
    for (auto &p: directory_iterator(path)) {
        std::cout << "loading image: " << p << std::endl;
        std::string name = p.path();
        int width, height, channels;
        unsigned char *image = stbi_load(name.c_str(),
                                         &width,
                                         &height,
                                         &channels,
                                         STBI_grey);

        mathRoutine::Image imageRes = mathRoutine::createImageFromUnsignedCharArray(image, width, height);
        result.emplace_back(imageRes);
        stbi_image_free(image);
    }

    return result;
}

void checkNonGPU(size_t iterations) {
    std::vector<mathRoutine::Image> images = prepareImages("data", 10);
    TotalGeneralizedVariation variation(std::move(images));

    float tau = 1 / (sqrtf(8)) / 4 / 8;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    mathRoutine::Image result = variation.solve(tau, lambda_tv, lambda_tgv, lambda_data, iterations);
    mathRoutine::writeImage(result, "result.png");
}

#else

void checkGPU(int argc, char **argv, size_t iterations, size_t amountOfImages) {
    float tau = 1 / (sqrtf(8)) / 4 /8 ;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    auto worker = GPUBasedTGV(argc, argv, 10);
    worker.init("data", amountOfImages);
    worker.start(iterations, tau, lambda_tv, lambda_tgv, lambda_data);
    worker.writeImage("result.png");
    worker.writePly("result.ply");
}

#endif


int main(int argc, char **argv) {
#ifdef CPU
    checkNonGPU(1000);
#else
    checkGPU(argc, argv, 0, 10); //DOES NOT WORK FOR NOW
#endif

    return 0;
}