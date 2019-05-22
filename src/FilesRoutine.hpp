//
// Created by roundedglint585 on 5/16/19.
//

#ifndef TGV_FILESROUTINES_HPP
#define TGV_FILESROUTINES_HPP

#include "StbInterfaceProxy.hpp"
#include "MathRoutine.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <fstream>
#include <tuple>

void writeImage(const std::string &name, const std::vector<float> &data, size_t height, size_t width) {
    auto result = data;
    unsigned char *image = new unsigned char[result.size()];
    ///Normalize
    for (auto &i: result) {
        if (i < 0.0f) {
            i = 0.0f;
        }
    }
    float max = 0.0f;
    for (auto &i: result) {
        if (max < i) {
            max = i;
        }
    }
    for (auto &i: result) {
        i = i / (max / 255.f);
    }
    ///
    for (size_t i = 0; i < result.size(); i++) {
        image[i] = (unsigned char) result[i];
    }
    stbi_write_png(name.c_str(), width, height, 1, image, width);
    delete[](image);
}

void writeImage(const mathRoutine::Image &result, std::string name) {
    unsigned char *image = std::move(mathRoutine::getArrayFromImage<unsigned char>(result).data());
    stbi_write_png(name.c_str(), result[0].size(), result.size(), 1, image, 1 * result.size());
    delete[] image;
}

//Width, Height, Data
std::tuple<size_t, size_t, std::vector<float>> readPFM(const std::string &name) {
    size_t height = 0, width = 0;
    std::vector<float> result;
    std::ifstream in(name.c_str());
    std::string read;
    for (size_t i = 0; i < 1; i++) {
        std::getline(in, read);
    }
    while (read[0] > '9' || read[0] < '0') {
        std::getline(in, read);
    }
    for (size_t i = 0; i < read.length(); i++) {
        if (read[i] == ' ') {
            width = std::stoi(read.substr(i, read.length()));
            height = std::stoi(read.substr(0, i));
            break;
        }
    }
    std::getline(in, read);
    result.reserve(height * width);
    float f;
    char b[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < height * width; i++) {
        in.read(b, 4);
        std::memcpy(&f, &b, sizeof(f));
        result.push_back(f);
    }
    return std::make_tuple(height, width, std::move(result));
}


void writePly(const std::string &name, const std::vector<float> &data, size_t height, size_t width, float xScaling,
              float yScaling) {
    std::ofstream out(name.c_str());
    out << "ply" << std::endl << "format ascii 1.0" << std::endl;
    out << "element vertex " << height * width << std::endl;
    out << "property float x" << std::endl << "property float y" << std::endl << "property float z" << std::endl;
    out << "element face " << ((height - 1) * (width - 1) * 2) << std::endl;
    out << "property list uint8 int32 vertex_indices" << std::endl << "end_header" << std::endl;
    auto result = data;
    //Normalize
    for (auto &i: result) {
        if (i < 0.0f) {
            i = 0.0f;
        }
    }
    //
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            out << xScaling * i << " " << yScaling * j << " " << result[j + i * width]
                << std::endl;
        }
    }

    for (size_t i = 0; i < height - 1; i++) {
        for (size_t j = 0; j < width - 1; j++) {
            auto a = j + i * width;
            auto b = j + 1 + i * width;
            auto c = j + width + i * width;
            auto d = j + width + 1 + i * width;
            out << "3 " << (int) a << " " << (int) c << " " << (int) b << std::endl;
            out << "3 " << (int) c << " " << (int) d << " " << (int) b << std::endl;
        }
    }
}

void writePFM(const std::string &name, size_t height, size_t width, const std::vector<float> &data) {
    std::ofstream out(name.c_str());
    out << "Pf" << std::endl << height << " " << width << std::endl << "-1" << std::endl; //Header
    out.close();
    out.open(name.c_str(), std::fstream::app | std::fstream::binary);
    for (auto &i: data) {
        out.write(reinterpret_cast<const char *>( &i), sizeof(float));
    }
}

#endif //TGV_FILESROUTINES_HPP
