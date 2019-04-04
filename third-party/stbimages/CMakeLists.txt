cmake_minimum_required(VERSION 3.1)

project(libstbimages)


set(HEADERS
        libstbimages/stb_image.h
        libstbimages/stb_image_write.h
        )

set(CMAKE_CXX_STANDARD 17)

set(LIBRARIES)


add_library(${PROJECT_NAME}  INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE libstbimages/)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
