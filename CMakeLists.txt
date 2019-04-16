cmake_minimum_required(VERSION 3.9)
project(TGV)

set(BUILD_RELEASE true CACHE BOOL "BUILD_RELEASE")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -fno-omit-frame-pointer\
")
add_subdirectory(third-party)


if(${BUILD_RELEASE})
    set(CMAKE_BUILD_TYPE Release)
    # convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
    # Она считывает все байты из файла src/cl/aplusb.cl (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле src/cl/aplusb_cl.h aplusb_kernel
    # Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл src/cl/aplusb_cl.h был перечислен среди исходников для компиляции при вызове add_executable
    convertIntoHeader(src/cl/gradient.cl src/cl/gradient.h gradient_kernel)
    convertIntoHeader(src/cl/epsilon.cl src/cl/epsilon.h epsilon_kernel)
    convertIntoHeader(src/cl/copy.cl src/cl/copy.h copy_kernel)
    convertIntoHeader(src/cl/mulMatrixOnConstant.cl src/cl/mulMatrixOnConstant.h mul_matrix_on_constant_kernel)
    convertIntoHeader(src/cl/project.cl src/cl/project.h project_kernel)
    convertIntoHeader(src/cl/sumOfMatrix.cl src/cl/sumOfMatrix.h sum_of_matrix_kernel)
    convertIntoHeader(src/cl/transpondedGradient.cl src/cl/transpondedGradient.h transponded_gradient_kernel)
    convertIntoHeader(src/cl/transpondedEpsilon.cl src/cl/transpondedEpsilon.h transponded_epsilon_kernel)
    convertIntoHeader(src/cl/calculateHist.cl src/cl/calculateHist.h calculate_hist_kernel)
    convertIntoHeader(src/cl/prox.cl src/cl/prox.h prox_kernel)

    add_executable(TGV main.cpp src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp  src/GPUBasedTotalGeneralizedVariation.cpp src/GPUBasedTotalGeneralizedVariation.hpp
            src/cl/epsilon.h src/cl/gradient.h src/cl/copy.h src/cl/mulMatrixOnConstant.h src/cl/project.h src/cl/sumOfMatrix.h src/cl/transpondedGradient.h src/cl/transpondedEpsilon.h src/cl/calculateHist.h
            src/cl/prox.h)
    target_link_libraries(TGV libclew libgpu libutils libstbimages stdc++fs)
else()
    find_package(GTest REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -fno-omit-frame-pointer\
    -lgtest\
    -lgtest_main \
    -lpthread \
    ")

    convertIntoHeader(src/cl/gradient.cl src/cl/gradient.h gradient_kernel)
    convertIntoHeader(src/cl/epsilon.cl src/cl/epsilon.h epsilon_kernel)
    convertIntoHeader(src/cl/copy.cl src/cl/copy.h copy_kernel)
    convertIntoHeader(src/cl/mulMatrixOnConstant.cl src/cl/mulMatrixOnConstant.h mul_matrix_on_constant_kernel)
    convertIntoHeader(src/cl/project.cl src/cl/project.h project_kernel)
    convertIntoHeader(src/cl/sumOfMatrix.cl src/cl/sumOfMatrix.h sum_of_matrix_kernel)
    convertIntoHeader(src/cl/transpondedGradient.cl src/cl/transpondedGradient.h transponded_gradient_kernel)
    convertIntoHeader(src/cl/transpondedEpsilon.cl src/cl/transpondedEpsilon.h transponded_epsilon_kernel)
    convertIntoHeader(src/cl/calculateHist.cl src/cl/calculateHist.h calculate_hist_kernel)
    convertIntoHeader(src/cl/prox.cl src/cl/prox.h prox_kernel)

    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable(runUnitTests tests/tests.cpp src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp src/cl/epsilon.h src/cl/gradient.h src/cl/copy.h src/cl/mulMatrixOnConstant.h src/cl/project.h src/cl/sumOfMatrix.h src/cl/transpondedGradient.h src/cl/transpondedEpsilon.h src/cl/calculateHist.h
            src/cl/prox.h)
    target_link_libraries(runUnitTests ${GTEST_LIBRARIES} pthread libclew libgpu libutils libstbimages stdc++fs)
endif()

