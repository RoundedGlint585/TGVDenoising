cmake_minimum_required(VERSION 3.9)
project(TGV)

set(BUILD_RELEASE true CACHE BOOL "BUILD_RELEASE")
set(BUILD_BENCHMARKS true CACHE BOOL "BUILD_BENCHMARKS")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -fno-omit-frame-pointer\
")
add_subdirectory(third-party)

# convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
# Она считывает все байты из файла src/kernels/aplusb.kernels (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле src/kernels/aplusb_cl.h aplusb_kernel
# Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл src/kernels/aplusb_cl.h был перечислен среди исходников для компиляции при вызове add_executable
convertIntoHeader(src/kernels/gradient.cl src/kernels/gradient.h gradient_kernel)
convertIntoHeader(src/kernels/epsilon.cl src/kernels/epsilon.h epsilon_kernel)
convertIntoHeader(src/kernels/copy.cl src/kernels/copy.h copy_kernel)
convertIntoHeader(src/kernels/mulMatrixOnConstant.cl src/kernels/mulMatrixOnConstant.h mul_matrix_on_constant_kernel)
convertIntoHeader(src/kernels/project.cl src/kernels/project.h project_kernel)
convertIntoHeader(src/kernels/sumOfMatrix.cl src/kernels/sumOfMatrix.h sum_of_matrix_kernel)
convertIntoHeader(src/kernels/transpondedGradient.cl src/kernels/transpondedGradient.h transponded_gradient_kernel)
convertIntoHeader(src/kernels/transpondedEpsilon.cl src/kernels/transpondedEpsilon.h transponded_epsilon_kernel)
convertIntoHeader(src/kernels/calculateHist.cl src/kernels/calculateHist.h calculate_hist_kernel)
convertIntoHeader(src/kernels/prox.cl src/kernels/prox.h prox_kernel)
convertIntoHeader(src/kernels/anorm.cl src/kernels/anorm.h anorm_kernel)


if (${BUILD_RELEASE})
    set(CMAKE_BUILD_TYPE Release)
    add_executable(TGV src/cli/main.cpp src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp src/GPUBasedTotalGeneralizedVariation.cpp src/GPUBasedTotalGeneralizedVariation.hpp
            src/kernels/epsilon.h src/kernels/gradient.h src/kernels/copy.h src/kernels/mulMatrixOnConstant.h src/kernels/project.h src/kernels/sumOfMatrix.h src/kernels/transpondedGradient.h src/kernels/transpondedEpsilon.h src/kernels/calculateHist.h
            src/kernels/prox.h src/kernels/anorm.h src/commonKernels.hpp)
    target_link_libraries(TGV libclew libgpu libutils libstbimages stdc++fs libcxxopts)
else()

    find_package(GTest REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -fno-omit-frame-pointer\
    -lgtest\
    -lgtest_main \
    -lpthread \
    ")
    convertIntoHeader(src/kernels/energy.cl src/kernels/energy.h energy_kernel)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable(runUnitTests tests/tests.cpp src/TotalGeneralizedVariation.cpp src/GPUBasedTotalGeneralizedVariation.cpp src/GPUBasedTotalGeneralizedVariation.hpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp src/kernels/epsilon.h src/kernels/gradient.h src/kernels/copy.h src/kernels/mulMatrixOnConstant.h src/kernels/project.h src/kernels/sumOfMatrix.h src/kernels/transpondedGradient.h src/kernels/transpondedEpsilon.h src/kernels/calculateHist.h
            src/kernels/prox.h src/kernels/anorm.h src/commonKernels.hpp src/kernels/energy.h)
    target_link_libraries(runUnitTests ${GTEST_LIBRARIES} pthread libclew libgpu libutils libstbimages stdc++fs)
endif()
