cmake_minimum_required(VERSION 3.2)
project(webgpu-demo)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(SDL2 REQUIRED)
find_package(wgpu-native REQUIRED)

file(GLOB_RECURSE WEBGPU_DEMO_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/source/*")
file(GLOB_RECURSE WEBGPU_DEMO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/*")

add_executable(webgpu-demo
	${WEBGPU_DEMO_HEADERS}
	${WEBGPU_DEMO_SOURCES}
)

target_link_libraries(webgpu-demo
	SDL2
	wgpu-native
)

target_include_directories(webgpu-demo PUBLIC
	${WGPU_NATIVE_INCLUDE_DIRECTORY}
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
