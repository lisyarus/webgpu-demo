cmake_minimum_required(VERSION 3.2)
project(webgpu-demo)

find_package(SDL2 REQUIRED)

find_library(WGPU_NATIVE_LIBRARY NAMES wgpu_native HINTS "${WGPU_NATIVE_ROOT}")
find_path(WGPU_NATIVE_INCLUDE_DIRECTORY NAMES webgpu.h HINTS "${WGPU_NATIVE_ROOT}")

add_executable(webgpu-demo source/main.cpp)
target_link_libraries(webgpu-demo SDL2 ${WGPU_NATIVE_LIBRARY})
target_include_directories(webgpu-demo PUBLIC ${WGPU_NATIVE_INCLUDE_DIRECTORY})
