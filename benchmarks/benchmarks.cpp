//
// Created by roundedglint585 on 5/14/19.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <filesystem>
#include "../src/GPUBasedTotalGeneralizedVariation.hpp"
#include "../src/TotalGeneralizedVariation.hpp"
#include "benchmark/benchmark.h"

constexpr size_t iter = 1000;

class GPUBase : public ::benchmark::Fixture {
public:
    GPUBase() : worker(0) {

    }

    void SetUp(const ::benchmark::State &st) {
        worker.init("data", 10);
    }

    void TearDown(const ::benchmark::State &) {

    }

    GPUBasedTGV worker;

};


BENCHMARK_DEFINE_F(GPUBase, Obj)(benchmark::State &state) {
    float tau = 1 / (sqrtf(8)) / 4 / 16;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    while (state.KeepRunning()) {
        for(int i = 0; i < iter; i++){
            worker.iteration(tau, lambda_tv, lambda_tgv, lambda_data);
        }
    }
}

BENCHMARK_REGISTER_F(GPUBase, Obj)->Iterations(1);

std::vector<mathRoutine::Image> prepareImages(std::string_view path, size_t amountOfImages) {
    std::vector<mathRoutine::Image> result;
    std::vector<std::vector<std::vector<float>>> transformed;
    using namespace std::filesystem;
    for (auto &p: directory_iterator(path)) {
        //std::cout << "loading image: " << p << std::endl;
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

class CPUBase : public ::benchmark::Fixture {
public:
    CPUBase() : worker(prepareImages("data", 10)) {

    }

    void SetUp(const ::benchmark::State &st) {
    }

    void TearDown(const ::benchmark::State &) {

    }

    TotalGeneralizedVariation worker;

};


BENCHMARK_DEFINE_F(CPUBase, Obj)(benchmark::State &state) {
    float tau = 1 / (sqrtf(8)) / 4 / 16;
    float lambda_data = 1.0;
    float lambda_tv = 1.0;
    float lambda_tgv = 1.0;
    lambda_tv /= lambda_data;
    lambda_tgv /= lambda_data;
    while (state.KeepRunning()) {
        for(int i = 0; i < iter; i++){
            worker.iteration(tau, lambda_tv, lambda_tgv, lambda_data);
        }
    }
}

BENCHMARK_REGISTER_F(CPUBase, Obj)->Iterations(1);


int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}