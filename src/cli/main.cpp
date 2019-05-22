#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <filesystem>
#include <cxxopts.hpp>
#include "../FilesRoutine.hpp"
#include "../TotalGeneralizedVariation.hpp"
#include "../GPUBasedTotalGeneralizedVariation.hpp"


std::vector<mathRoutine::Image> prepareImagesCPU(std::string_view path, size_t amountOfImages) {
    std::vector<mathRoutine::Image> result;
    std::vector<std::vector<std::vector<float>>> transformed;
    using namespace std::filesystem;
    for (auto &p: directory_iterator(path)) {
        std::string name = p.path();
        int width, height;
        auto image = readPFM(name.c_str());
        width = std::get<0>(image);
        height = std::get<1>(image);
        mathRoutine::Image imageRes = mathRoutine::createImageFromUnsignedCharArray(std::get<2>(image).data(), width,
                                                                                    height);
        result.emplace_back(imageRes);
    }

    return result;
}

std::tuple<size_t, size_t, std::vector<float>>
prepareImagesGPU(std::string_view path, size_t amountOfImages) {
    using namespace std::filesystem;
    size_t totalSize = 0;
    size_t totalAmountOfImages = 0;
    int width = 0, height = 0;
    std::vector<float> observations;
    std::vector<float> image;
    for (auto &p: directory_iterator(path)) {
        std::string name = p.path();
        auto res = readPFM(name.c_str());
        width = std::get<0>(res);
        height = std::get<1>(res);
        for (size_t i = 0; i < static_cast<size_t>(width) * height; i++) {
            observations.emplace_back(std::get<2>(res)[i]);
        }
        totalAmountOfImages++;
        totalSize += width * height;
    }
    /// Preprocessing
    float mean = 0.f;
    size_t amount = 0;
    for (auto &i: observations) {
        if (i >= -32766.f) {
            mean += i;
            amount++;
        }
    }
    mean = mean / amount;
    ///
    if (totalAmountOfImages <= amountOfImages) {
        return std::make_tuple(width, height, observations);
    } else {
        std::vector<float> selectedObservation = std::vector(width * height * amountOfImages, 0.f);

        for (size_t i = 0; i < static_cast<size_t>(height); i++) {
            for (size_t j = 0; j < static_cast<size_t>(width); j++) {
                std::vector<float> allForPixel;
                for (size_t k = 0; k < totalAmountOfImages; k++) {
                    allForPixel.emplace_back(observations[j + i * width + k * width * height]);
                }
                /// Preprocessing
                std::sort(allForPixel.begin(), allForPixel.end());
                if (allForPixel[allForPixel.size() - 1] < -32766.0f) {
                    allForPixel = std::vector<float>(totalAmountOfImages, mean);

                } else {
                    for (size_t i = 0; i < allForPixel.size(); i++) {
                        if (allForPixel[i] < -32766.0f) {
                            allForPixel.erase(allForPixel.begin() + i);
                            i--;
                        }
                    }
                    for (size_t i = 0; i < (totalAmountOfImages - allForPixel.size()); i++) {
                        allForPixel.push_back(allForPixel[i % allForPixel.size()]);
                    }
                    std::sort(allForPixel.begin(), allForPixel.end());
                }
                ///

                size_t left = (totalAmountOfImages - amountOfImages) / 2;
                for (size_t k = 0; k < amountOfImages; k++) {
                    selectedObservation[j + i * width + k * width * height] = allForPixel[k + left];
                }
            }
        }
        return std::make_tuple(width, height,
                               selectedObservation);
    }

}

void CPU(size_t iterations, const std::string &path, const std::string &resultFileName) {
    std::vector<mathRoutine::Image> images = prepareImagesCPU(path.c_str(), 10);
    TotalGeneralizedVariation variation(std::move(images));
    float tau = 1 / (sqrtf(8)) / 4 / 8;
    float lambda_data = 1.0;
    float lambda_tv = 1.0 / 3.0;
    float lambda_tgv = 2.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    for (size_t i = 0; i < iterations; i++) {
        if (i % 100 == 0) {
            std::cout << "\rIteration # " << i << " out of " << iterations;
            std::cout.flush();
        }
        variation.iteration(tau, lambda_tv, lambda_tgv, lambda_data);
    }
    std::cout << std::endl;
    mathRoutine::Image result = variation.getImage();
    auto linedUp = mathRoutine::getArrayFromImage<float>(result);
    writeImage(result, resultFileName + ".png");
    writePFM(resultFileName + ".png", result.size(), result[0].size(), linedUp);
    writePly(resultFileName + ".ply", linedUp, result.size(), result[0].size(), 1.f, 1.f);
}


void GPU(size_t index, size_t iterations, size_t amountOfImages, const std::string &path,
         const std::string &resultFileName) {
    float tau = 1 / (sqrtf(8)) / 4 / 8;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    auto worker = GPUBasedTGV(index);
    auto data = prepareImagesGPU(path, amountOfImages);
    size_t imageSize = std::get<0>(data) * std::get<1>(data);
    std::vector<float> image(imageSize);
    for (size_t i = 0; i < imageSize; i++) {
        image[i] = std::get<2>(data)[i];
    }
    worker.init(amountOfImages, std::get<0>(data), std::get<1>(data), image, std::get<2>(data));
    std::cout << "Reading files from: " << path << std::endl;
    for (size_t i = 0; i < iterations; i++) {
        if (i % 100 == 0) {
            std::cout << "\rIteration # " << i << " out of " << iterations;
            std::cout.flush();
        }
        worker.iteration(tau, lambda_tv, lambda_tgv, lambda_data);
    }
    std::cout << std::endl;
    auto result = worker.getImage();
    writeImage(resultFileName + ".png", result, worker.getHeight(), worker.getWidth());
    writePFM(resultFileName + ".png", worker.getHeight(), worker.getWidth(), result);
    writePly(resultFileName + ".ply", result, worker.getHeight(), worker.getWidth(), 1.f, 1.f);
}


int main(int argc, char **argv) {
    cxxopts::Options options("TGV denoising", "Image denoising based on TGV");
    options.add_options()
            ("c", "Use CPU")
            ("g", "Use GPU")
            ("n", "Amount of iterations", cxxopts::value<size_t>()->default_value("1000"))
            ("p", "Path to data", cxxopts::value<std::string>()->default_value("data"))
            ("a", "GPU Device number, if gpu used", cxxopts::value<size_t>()->default_value("0"))
            ("r", "Result files name(ply + png)", cxxopts::value<std::string>()->default_value("result"))
            ("i", "Amount of images from whole data set", cxxopts::value<size_t>()->default_value("10"))
            ("scaleX", "Scale on x for ply file", cxxopts::value<float>()->default_value("1.0"))
            ("scaleY", "Scale on y for ply file", cxxopts::value<float>()->default_value("1.0"));
    auto result = options.parse(argc, argv);

    if (result["g"].as<bool>() && result["c"].as<bool>()) {
        std::cerr << "Choose GPU or CPU, not both" << std::endl;
        return 1;
    }

    if (result["c"].as<bool>()) {
        std::cout << "Using CPU" << std::endl;
        std::cout << "Amount of iterations: " << result["n"].as<size_t>() << std::endl;
        CPU(result["n"].as<size_t>(), result["p"].as<std::string>(), result["r"].as<std::string>());
    } else {
        std::cout << "Using GPU" << std::endl;
        std::cout << "Amount of iterations: " << result["n"].as<size_t>() << std::endl;
        GPU(result["a"].as<std::size_t>(), result["n"].as<size_t>(), result["i"].as<size_t>(),
            result["p"].as<std::string>(), result["r"].as<std::string>()); //DOES NOT WORK FOR NOW
        std::cout << "Result saved as: " << result["r"].as<std::string>() << ".png and "
                  << result["r"].as<std::string>() << ".ply" << std::endl;
    }


    return 0;
}