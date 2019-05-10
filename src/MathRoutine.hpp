//
// Created by roundedglint585 on 3/19/19.
//

#ifndef TGV_MATHROUTINE_HPP
#define TGV_MATHROUTINE_HPP

#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <stb_image.h>
#include <stb_image_write.h>
#include <fstream>

namespace mathRoutine {
    constexpr float eps = 0.001;
    using Image = std::vector<std::vector<float>>;
    using Gradient = std::vector<std::vector<std::array<float, 2>>>;
    using Epsilon = std::vector<std::vector<std::array<float, 4>>>;

    static Image createImageFromUnsignedCharArray(const unsigned char *image, size_t width, size_t height);

    static unsigned char *getArrayFromImage(std::size_t *width, std::size_t *height, const Image &image);

    static void writeImage(mathRoutine::Image result, std::string name);

    static Gradient calculateGradient(const Image &image);

    static Epsilon calculateEpsilon(const Gradient &gradient);

    static Image calculateTranspondedGradient(const Gradient &gradient);

    static Gradient calculateTranspondedEpsilon(const Epsilon &epsilon);

    template<typename Matrix>
    static Image anorm(const Matrix &matrix);

    template<typename Matrix>
    static Matrix project(const Matrix &matrix, float r);

    static Image sumOfImage(const Image &image1, const Image &image2);

    static Image mulImageOnConstant(const Image &image, float k);

    template<typename Matrix>
    static Matrix sumOfMatrix(const Matrix &matrix1, const Matrix &matrix2);

    template<typename Matrix>
    static Matrix mulMatrixOnConstant(const Matrix &matrix1, float k);

    static Image
    operator+(const Image &image1, const Image &image2);

    static Image
    operator*(const Image &image1, float r);

    static Image
    operator*(float r, const Image &image1);

    template<typename Matrix>
    static Matrix operator+(const Matrix &matrix1, const Matrix &matrix2);

    template<typename Matrix>
    static Matrix operator*(const Matrix &matrix1, float r);

    template<typename Matrix>
    static Matrix operator*(float r, const Matrix &matrix1);

    void writeImage(mathRoutine::Image result, std::string name) {
        size_t width, height;
        unsigned char *image = mathRoutine::getArrayFromImage(&width, &height, result);
        stbi_write_png(name.c_str(), result[0].size(), result.size(), 1, image, 1 * result.size());
    }

    static Image createImageFromUnsignedCharArray(const unsigned char *image, size_t width, size_t height) {
        Image result(height, std::vector<float>(width, 0));
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i][j] = (float) image[j + width * i];
            }
        }
        return result;
    }

    static unsigned char *getArrayFromImage(std::size_t *width, std::size_t *height, const Image &image) {
        unsigned char *result = new unsigned char[image.size() * image[0].size()];
        *height = image.size();
        *width = image[0].size();
        for (size_t i = 0; i < image.size(); i++) {
            for (size_t j = 0; j < image[0].size(); j++) {
                result[j + image[0].size() * i] = (unsigned char) image[i][j];
            }
        }
        return result;
    }

    static Gradient calculateGradient(const Image &image) {
        Gradient result(image.size(), std::vector<std::array<float, 2>>(image[0].size()));
        size_t height = image.size();
        size_t width = image[0].size();
        for (size_t i = 0; i < height; i++) {
            result[i].resize(image[0].size());
            for (size_t j = 1; j < width; j++) {
                result[i][j - 1][0] = image[i][j] - image[i][j - 1];
            }
        }
        for (size_t i = 1; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i - 1][j][1] = image[i][j] - image[i - 1][j];
            }
        }
        return result;
    }

    static Epsilon calculateEpsilon(const Gradient &gradient) {
        Epsilon result(gradient.size(), std::vector<std::array<float, 4>>(gradient[0].size()));
        size_t height = gradient.size();
        size_t width = gradient[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 1; j < width; j++) {
                result[i][j - 1][0] = gradient[i][j][0] - gradient[i][j - 1][0];
                result[i][j - 1][2] = gradient[i][j][1] - gradient[i][j - 1][1];
            }
        }
        for (size_t i = 1; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i - 1][j][1] = gradient[i][j][0] - gradient[i - 1][j][0];
                result[i - 1][j][3] = gradient[i][j][1] - gradient[i - 1][j][1];
            }
        }
        return result;
    }

    static Image calculateTranspondedGradient(const Gradient &gradient) {
        Image result = Image(gradient.size(), std::vector<float>(gradient[0].size(), 0));
        size_t height = gradient.size();
        size_t width = gradient[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i][j] -= gradient[i][j][0];
                result[i][j] -= gradient[i][j][1];
            }
        }
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width - 1; j++) {
                result[i][j + 1] += gradient[i][j][0]; //result[2,2] += gradient[2,1]
            }
        }
        for (size_t i = 0; i < height - 1; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i + 1][j] += gradient[i][j][1];
            }
        }
        return result;
    }

    static Gradient calculateTranspondedEpsilon(const Epsilon &epsilon) {
        Gradient result(epsilon.size(), std::vector<std::array<float, 2>>(epsilon[0].size()));
        size_t height = epsilon.size();
        size_t width = epsilon[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i][j][0] -= epsilon[i][j][0];
                result[i][j][0] -= epsilon[i][j][1];
                result[i][j][1] -= epsilon[i][j][2];
                result[i][j][1] -= epsilon[i][j][3];
            }
        }
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width - 1; j++) {
                result[i][j + 1][0] += epsilon[i][j][0];
                result[i][j + 1][1] += epsilon[i][j][2];
            }
        }
        for (size_t i = 0; i < height - 1; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i + 1][j][0] += epsilon[i][j][1];
                result[i + 1][j][1] += epsilon[i][j][3];
            }
        }
        return result;
    }

    template<typename Matrix>
    static Image anorm(const Matrix &matrix) {
        Image result = Image(matrix.size(), std::vector<float>(matrix[0].size(), 0));
        size_t height = matrix.size();
        size_t width = matrix[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                for (auto &k: matrix[i][j]) {
                    result[i][j] += k * k;
                }
                result[i][j] = sqrtf(result[i][j]);
            }
        }
        return result;
    }

    template<typename Matrix>
    static Matrix project(const Matrix &matrix, float r) {
        Matrix result = matrix;
        size_t height = matrix.size();
        size_t width = matrix[0].size();
        Image normed = anorm(result);
        for(auto& i: normed){
            for(auto& j: i){
                j /= r;
                if(j < eps){
                    j = 1.f;
                }
            }
        }
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                for (auto &k: result[i][j]) {
                    k /= normed[i][j];
                }
            }
        }
        return result;
    }


    static Image sumOfImage(const Image &image1, const Image &image2) {
        Image result = image1;
        size_t height = image1.size();
        size_t width = image1[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i][j] += image2[i][j];
            }
        }
        return result;
    }

    static Image mulImageOnConstant(const Image &image, float k) {
        Image result = image;
        size_t height = image.size();
        size_t width = image[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                result[i][j] *= k;
            }
        }
        return result;
    }

    template<typename Matrix>
    static Matrix sumOfMatrix(const Matrix &matrix1, const Matrix &matrix2) {
        Matrix result = matrix1;
        size_t height = matrix1.size();
        size_t width = matrix1[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                for (size_t k = 0; k < matrix2[i][j].size(); k++) {
                    result[i][j][k] += matrix2[i][j][k];
                }
            }
        }
        return result;
    }

    template<typename Matrix>
    static Matrix mulMatrixOnConstant(const Matrix &matrix1, float k) {
        Matrix result = matrix1;
        size_t height = matrix1.size();
        size_t width = matrix1[0].size();
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                for (auto &p: result[i][j]) {
                    p *= k;
                }
            }
        }
        return result;
    }


    static Image
    operator+(const Image &image1, const Image &image2) {
        Image result = image1;
        for (size_t i = 0; i < image1.size(); i++) {
            for (size_t j = 0; j < image1[0].size(); j++) {
                result[i][j] += image2[i][j];
            }
        }
        return result;
    }

    static Image
    operator*(const Image &image1, float r) {
        Image result = image1;
        for (size_t i = 0; i < image1.size(); i++) {
            for (size_t j = 0; j < image1[0].size(); j++) {
                result[i][j] *= r;
            }
        }
        return result;
    }

    static Image
    operator*(float r, const Image &image1) {
        return image1 * r;
    }

    template<typename Matrix>
    static Matrix operator+(const Matrix &matrix1, const Matrix &matrix2) {
        Matrix result = matrix1;
        for (size_t i = 0; i < matrix2.size(); i++) {
            for (size_t j = 0; j < matrix2[0].size(); j++) {
                for (size_t k = 0; k < matrix2[i][j].size(); k++) {
                    result[i][j][k] += matrix2[i][j][k];
                }
            }
        }
        return result;
    }

    template<typename Matrix>
    static Matrix operator*(const Matrix &matrix1, float r) {
        Matrix result = matrix1;
        for (size_t i = 0; i < matrix1.size(); i++) {
            for (size_t j = 0; j < matrix1[0].size(); j++) {
                for (auto &p: result[i][j]) {
                    p *= r;
                }
            }
        }
        return result;
    }

    template<typename Matrix>
    static Matrix operator*(float r, const Matrix &matrix1) {
        return matrix1 * r;
    }


}
#endif //TGV_MATHROUTINE_HPP
