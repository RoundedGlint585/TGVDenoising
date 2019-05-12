cmake_minimum_required(VERSION 3.1)

project(libcxxopts)


set(HEADERS
        libcxxopts/cxxopts.hpp)

set(CMAKE_CXX_STANDARD 17)

set(LIBRARIES)


add_library(${PROJECT_NAME}  INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE libcxxopts/)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
