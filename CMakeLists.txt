cmake_minimum_required(VERSION 3.9)
project(TGV)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -fno-omit-frame-pointer\
")
add_subdirectory(third-party)


if(!test)
    # convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
    # Она считывает все байты из файла src/cl/aplusb.cl (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле src/cl/aplusb_cl.h aplusb_kernel
    # Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл src/cl/aplusb_cl.h был перечислен среди исходников для компиляции при вызове add_executable
    convertIntoHeader(src/cl/gradient.cl src/cl/gradient.h gradient_kernel)
    convertIntoHeader(src/cl/epsilon.cl src/cl/epsilon.h epsilon_kernel)
    convertIntoHeader(src/cl/copy.cl src/cl/copy.h copy_kernel)

    add_executable(TGV main.cpp  src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp  src/GPUBasedTotalGeneralizedVariation.cpp src/GPUBasedTotalGeneralizedVariation.hpp src/cl/epsilon.h src/cl/gradient.h src/cl/copy.h)
    target_link_libraries(TGV libclew libgpu libutils libstbimages stdc++fs)
else()
    find_package(GTest REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -fno-omit-frame-pointer\
    -lgtest\
    -lgtest_main \
    -lpthread \
    ")

    include_directories(${GTEST_INCLUDE_DIRS})


    add_executable(runUnitTests tests/tests.cpp src/TotalGeneralizedVariation.cpp src/TotalGeneralizedVariation.hpp src/MathRoutine.hpp )
    target_link_libraries(runUnitTests libstbimages ${GTEST_LIBRARIES} pthread)
endif()

