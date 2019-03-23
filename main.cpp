#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include "third-party/stb_image.h"
#include "src/MathRoutine.hpp"
int main() {
    int width, height, channels;
    unsigned char *image = stbi_load("res/result.png",
                                     &width,
                                     &height,
                                     &channels,
                                     STBI_grey);

    mathRoutine::Image imageInMatrix = mathRoutine::createImageFromUnsignedCharArray(image, height, channels);

    std::size_t returnedWidth, returnedHeight;
    return 0;
}