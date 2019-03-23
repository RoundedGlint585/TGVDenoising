cmake_minimum_required(VERSION 3.13)
project(TGV)

option(test "Build all tests." ON) # Makes boolean 'test' available.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -Werror\
    -fsanitize=address -fno-omit-frame-pointer\
")


add_executable(TGV main.cpp third-party/stb_image.h third-party/stb_image_write.h src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp)


##TESTING
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
enable_testing()

add_executable(runUnitTests tests/tests.cpp third-party/stb_image.h third-party/stb_image_write.h src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp)
target_link_libraries(runUnitTests gtest gtest_main)


