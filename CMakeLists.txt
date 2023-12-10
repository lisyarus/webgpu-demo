cmake_minimum_required(VERSION 3.2)
project(webgpu-demo)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)

find_package(SDL2 REQUIRED)
find_package(wgpu-native REQUIRED)

file(GLOB_RECURSE WEBGPU_DEMO_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/source/*")
file(GLOB_RECURSE WEBGPU_DEMO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/*")

add_executable(webgpu-demo
	${WEBGPU_DEMO_HEADERS}
	${WEBGPU_DEMO_SOURCES}
)

target_link_libraries(webgpu-demo
	SDL2::SDL2
	wgpu-native
)

target_include_directories(webgpu-demo PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/glm"
	"${CMAKE_CURRENT_SOURCE_DIR}/rapidjson/include"
)

target_compile_definitions(webgpu-demo PUBLIC
	-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
	set_source_files_properties("source/sdl_wgpu.c" PROPERTIES COMPILE_FLAGS "-x objective-c")
	target_link_libraries(webgpu-demo
		"-framework QuartzCore"
		"-framework Cocoa"
		"-framework Metal"
	)
endif()
