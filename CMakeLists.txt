cmake_minimum_required(VERSION 3.3)
project(Hough)

find_package(Armadillo REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

configure_file(test.png ${CMAKE_CURRENT_BINARY_DIR}/test.png COPYONLY)
configure_file(test2.png ${CMAKE_CURRENT_BINARY_DIR}/test2.png COPYONLY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -g -march=native -ftree-vectorize -fwhole-program -Wall -Werror")

set(SOURCE_FILES main.cpp edge_detect_operators.h)
add_executable(Hough ${SOURCE_FILES})
target_link_libraries(Hough ${ARMADILLO_LIBRARIES} ${OpenCV_LIBS})