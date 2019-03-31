cmake_minimum_required(VERSION 3.9)
project(TGV)

option(test "Build all tests." ON) # Makes boolean 'test' available.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -fsanitize=address\
    -Wall\
    -fno-omit-frame-pointer\
")
add_subdirectory(third-party)

set(test ON)

add_executable(TGV main.cpp src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp)
target_link_libraries(TGV libclew libgpu libutils libstbimages)

if(test)
    find_package(GTest REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})
    enable_testing()

    add_executable(runUnitTests tests/tests.cpp src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp)
    target_link_libraries(runUnitTests gtest gtest_main libstbimages)
endif()

