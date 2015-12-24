cmake_minimum_required(VERSION 3.3)
project(Hough)

find_package(Armadillo REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -O2 -g -march=native -ftree-vectorize -Wall -Werror -Wno-unused -DARMA_NO_DEBUG")

set(SOURCE_FILES edge_detection.h hough_transform.h hough_transform.cpp main.cpp image_io.h)
add_executable(Hough ${SOURCE_FILES})
target_link_libraries(Hough ${ARMADILLO_LIBRARIES} ${OpenCV_LIBS})
