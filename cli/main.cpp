#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <filesystem>
#include <cxxopts.hpp>
#include "../src/TotalGeneralizedVariation.hpp"
#include "../src/GPUBasedTotalGeneralizedVariation.hpp"


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

void CPU(size_t iterations, const std::string& path, const std::string& resultFileName) {
    std::vector<mathRoutine::Image> images = prepareImages(path.c_str(), 10);
    TotalGeneralizedVariation variation(std::move(images));
    float tau = 1 / (sqrtf(8)) / 4 / 8;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    for(size_t i = 0; i < iterations; i++) {
        if(i % 100 == 0){
            std::cout << "\rIteration # " << i << " out of " << iterations;
            std::cout.flush();
        }
        variation.iteration(tau, lambda_tv, lambda_tgv, lambda_data);
    }
    std::cout << std::endl;
    mathRoutine::Image result = variation.getImage();
    mathRoutine::writeImage(result, resultFileName + ".png");
}


void GPU(size_t index, size_t iterations, size_t amountOfImages, const std::string& path, const std::string& resultFileName) {
    float tau = 1 / (sqrtf(8))/4;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    auto worker = GPUBasedTGV(index);
    worker.init(path.c_str(), amountOfImages);
    std::cout << "Reading files from: " << path << std::endl;
    for(size_t i = 0; i < iterations; i++) {
        if(i % 100 == 0){
            std::cout << "\rIteration # " << i << " out of " << iterations;
            std::cout.flush();
        }
        worker.iteration(tau, lambda_tv, lambda_tgv, lambda_data);
    }
    std::cout << std::endl;
    worker.writeImage(resultFileName + ".png");
    worker.writePly(resultFileName + ".ply");
}




int main(int argc, char **argv) {
    cxxopts::Options options("TGV denoising", "Image denoising based on TGV");
    options.add_options()
            ("c", "Use CPU")
            ("g", "Use GPU")
            ("n", "Amount of iterations", cxxopts::value<size_t>() -> default_value("1000"))
            ("p", "Path to data", cxxopts::value<std::string>()->default_value("data"))
            ("a", "GPU Device number, if gpu used", cxxopts::value<size_t>()->default_value("0"))
            ("r", "Result files name(ply + png)", cxxopts::value<std::string>()->default_value("result"))
            ("i", "Amount of images from whole data set", cxxopts::value<size_t>()->default_value("10"));
    auto result = options.parse(argc, argv);

    if (result["g"].as<bool>() && result["c"].as<bool>()) {
        std::cerr << "Choose GPU or CPU, not both" << std::endl;
        return 1;
    }

    if (result["c"].as<bool>()) {
        std::cout << "Using CPU" << std::endl;
        std::cout << "Amount of iterations: " <<  result["n"].as<size_t>() << std::endl;
        CPU(result["n"].as<size_t>(), result["p"].as<std::string>(), result["r"].as<std::string>());
    } else {
        std::cout << "Using GPU" << std::endl;
        std::cout << "Amount of iterations: " <<  result["n"].as<size_t>() << std::endl;
        GPU(result["a"].as<size_t>(), result["n"].as<size_t>(), result["i"].as<size_t>(), result["p"].as<std::string>(), result["r"].as<std::string>()); //DOES NOT WORK FOR NOW
        std::cout << "Result saved as: " <<result["r"].as<std::string>() <<".png and " << result["r"].as<std::string>() << ".ply" << std::endl;
    }


    return 0;
}