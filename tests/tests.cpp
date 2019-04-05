//
// Created by roundedglint585 on 3/19/19.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <gtest/gtest.h>
#include "../src/MathRoutine.hpp"

TEST(imageTest, readImageFromFile) {
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

TEST(imageTest, writeAndLoad) {
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

TEST(imageTest, calculateGradient) {
    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Gradient correct = {{{0.f,   0},     {0,     15.f},  {0,      0},     {0,      0},      {0, 0}},
                                     {{15.f,  15.f},  {-15.f, -15.f}, {0,      0},     {0,      195.f},  {0, 0}},
                                     {{-15.f, -15.f}, {0,     0},     {195.f,  0},     {-195.f, 4.f},    {0, 0}},
                                     {{0,     0},     {0,     195.f}, {199.f,  199.f}, {-199.f, -197.f}, {0, 0}},
                                     {{195.f, 0},     {4.0f,  0},     {-197.f, 0},     {-2.f,   0},      {0, 0}}};
    ASSERT_TRUE(gradient == correct) << "Gradient calculation error";
}

TEST(imageTest, calculateEpsilon) {
    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Epsilon correctEpsilon = {{{0,      15.f,  15.f,  15.f},  {0,      -15.f, -15.f, -30.f},  {0,      0,      0,      0},      {0,     0,      0,      195.f},  {0, 0, 0, 0}},
                                           {{-30.f,  -30.f, -30.f, -30.f}, {15.f,   15.f,  15.f,  15.f},   {0,      195.f,  195.f,  0},      {0,     -195.f, -195.f, -191.f}, {0, 0, 0, 0}},
                                           {{15.f,   15.f,  15.f,  15.f},  {195.f,  0,     0,     195.f},  {-390.f, 4.f,    4.f,    199.f},  {195.f, -4.f,   -4.f,   -201.f}, {0, 0, 0, 0}},
                                           {{0,      195.f, 195.f, 0},     {199.f,  4.f,   4.f,   -195.f}, {-398.f, -396.f, -396.f, -199.f}, {199.f, 197.f,  197.f,  197.f},  {0, 0, 0, 0}},
                                           {{-191.f, 0,     0,     0},     {-201.f, 0,     0,     0},      {195.f,  0,      0,      0},      {2.f,   0,      0,      0},      {0, 0, 0, 0},}};
    ASSERT_TRUE(epsilon == correctEpsilon) << "Epsilon calculation error";
}

TEST(imageTest, calculateTranspondedGradient) {
    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Image transpondedGradient = mathRoutine::calculateTranspondedGradient(gradient);
    mathRoutine::Image correctTransponded = {{0,      -15.f,  0,      0,      0},
                                             {-30.f,  60.f,   -15.f,  -195.f, 0},
                                             {45.f,   -30.f,  -195.f, 581.f,  -195.f},
                                             {-15.f,  -195.f, -398.f, 599.f,  -199.f},
                                             {-195.f, 386.f,  400.f,  -392.f, -2.f}};
    ASSERT_TRUE(transpondedGradient == correctTransponded) << "Transpoded Gradient calculation error";
}

TEST(imageTest, calculateTranspondedEpsilon) {
    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Gradient transpondedEpsilon = mathRoutine::calculateTranspondedEpsilon(epsilon);
    mathRoutine::Gradient correctTransoponded = {{{-15.f,  -30.f},  {15.f,   60.f},   {0,      -15.f},  {0,      -195.f}, {0,     0}},
                                                 {{75.f,   75.f},   {-75.f,  -90.f},  {-180.f, -180.f}, {195.f,  776.f},  {0,     -195.f}},
                                                 {{-60.f,  -60.f},  {-165.f, -165.f}, {776.f,  -203.f}, {-776.f, 18.f},   {195.f, -4.f}},
                                                 {{-180.f, -180.f}, {-203.f, 581.f},  {997.f,  798.f},  {-798.f, -991.f}, {199.f, 197.f}},
                                                 {{386.f,  0},      {14.f,   -195.f}, {-792.f, -199.f}, {390.f,  197.f},  {2.f,   0}}};
    ASSERT_TRUE(transpondedEpsilon == correctTransoponded) << "transponded epsilon calculation error";
}

TEST(imageTest, calculateAnorm) {
    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Image normalizedGradient = mathRoutine::anorm(gradient);
    mathRoutine::Image correctNormalizedGradient = {{0,          15.f,       0,         0,          0,},
                                                    {21.213203f, 21.213203f, 0,         195.f,      0},
                                                    {21.213203f, 0,          195.f,     195.04102f, 0},
                                                    {0,          195.f,      281.4285f, 280.01785f, 0},
                                                    {195.f,      4.f,        197.f,     2.f,        0,}};
    for (size_t i = 0; i < normalizedGradient.size(); i++) {
        for (size_t j = 0; j < normalizedGradient[0].size(); j++) {
            ASSERT_NEAR(normalizedGradient[i][j], correctNormalizedGradient[i][j], mathRoutine::eps);
        }
    }
    mathRoutine::Image normalizedEpsilon = mathRoutine::anorm(epsilon);
    mathRoutine::Image correctNormalizedEpsilon = {{25.9808, 36.7423, 0,       195,     0},
                                                   {60,      30,      275.772, 335.456, 0},
                                                   {30,      275.772, 437.873, 280.104, 0},
                                                   {275.772, 278.672, 715.288, 395.004, 0},
                                                   {191,     201,     195,     2,       0}};
    for (size_t i = 0; i < normalizedEpsilon.size(); i++) {
        for (size_t j = 0; j < normalizedEpsilon[0].size(); j++) {
            ASSERT_NEAR(normalizedEpsilon[i][j], correctNormalizedEpsilon[i][j], mathRoutine::eps);
        }
    }
}

TEST(imageTest, projectOfMatrixTest) {
    mathRoutine::Image imageInMatrix = {{1,  1,   1,   1,   1},
                                        {1,  16,  1,   1,   1},
                                        {16, 1,   1,   196, 1},
                                        {1,  1,   1,   200, 1},
                                        {1,  196, 200, 3,   1}};
    mathRoutine::Gradient gradient = mathRoutine::calculateGradient(imageInMatrix);
    mathRoutine::Epsilon epsilon = mathRoutine::calculateEpsilon(gradient);
    mathRoutine::Gradient projectedGradient = mathRoutine::project(gradient, 2);
    mathRoutine::Gradient correctProjectedGradient = {{{0,         0},         {0,         2.f},       {0,        0},        {0,         0},         {0, 0},},
                                                      {{1.41421f,  1.41421f},  {-1.41421f, -1.41421f}, {0,        0},        {0,         2.f},       {0, 0},},
                                                      {{-1.41421f, -1.41421f}, {0,         0},         {2.f,      0},        {-1.99958f, 0.041017f}, {0, 0},},
                                                      {{0,         0},         {0,         2.f},       {1.41421f, 1.41421f}, {-1.42134f, -1.40705f}, {0, 0},},
                                                      {{2.f,       0},         {2.f,       0},         {-2.f,     0},        {-2.f,      0},         {0, 0},}};
    for (size_t i = 0; i < projectedGradient.size(); i++) {
        for (size_t j = 0; j < projectedGradient[0].size(); j++) {
            ASSERT_NEAR(projectedGradient[i][j][0], correctProjectedGradient[i][j][0], mathRoutine::eps);
            ASSERT_NEAR(projectedGradient[i][j][1], correctProjectedGradient[i][j][1], mathRoutine::eps);
        }
    }
}

TEST(imageTest, sumOfImages) {
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
