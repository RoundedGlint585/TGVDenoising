#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include "src/MathRoutine.hpp"
#include "src/TotalGeneralizedVariation.hpp"
std::vector<mathRoutine::Image> prepareImages(size_t amount) {
    std::vector<mathRoutine::Image> result;
    for (size_t i = 0; i < amount; i++) {
        int width, height, channels;
        std::string name("data/house_" + std::to_string(i) + ".png");
        std::cout << "loaded: " << name << std::endl;
        unsigned char *image = stbi_load(name.c_str(),
                                         &width,
                                         &height,
                                         &channels,
                                         STBI_grey);
        mathRoutine::Image imageRes = mathRoutine::createImageFromUnsignedCharArray(image, width, height);
        result.push_back(imageRes);
        stbi_image_free(image);
    }
    return result;
}



int main() {
    std::vector<mathRoutine::Image> images = prepareImages(10);
    TotalGeneralizedVariation variation(std::move(images));

    float tau = 1 / (sqrtf(8)) / 4 / 8;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    mathRoutine::Image result = variation.solve(tau, lambda_tv, lambda_tgv, lambda_data, 400);
    mathRoutine::writeImage(result, "result.png");
    return 0;
}