# About

A demo WebGPU renderer written on streams at twitch.tv/lisyarus, mostly for me to get familiar with WebGPU.

It uses wgpu-native WebGPU implementation, and SDL2 to create a window to render to.

# Links

* Trello board with ongoing tasks: https://trello.com/b/DlDOU1Hs/webgpu-demo
* Stream recordings: https://www.youtube.com/@lisyarus

# Building

To build this project, you need
* [CMake](https://cmake.org)
* [SDL2](https://www.libsdl.org/) (you can probably install it via your system's package manager)
* [wgpu-native](https://github.com/gfx-rs/wgpu-native)

To install wgpu-native, download [some release archive](https://github.com/gfx-rs/wgpu-native/releases) for your platform, and unpack it somewhere. This project was built with the [v0.18.1.3](https://github.com/gfx-rs/wgpu-native/releases/tag/v0.18.1.3) release, and might not work with other version.

Then, follow the usual steps for building something with CMake:
* Create a build directory
* In the build directory, run `cmake <path-to-webgpu-demo-source> -DWGPU_NATIVE_ROOT=<path-to-unpacked-wgpu-native>`
* Build the project: `cmake --build .`

Note that building for MacOS doesn't fully work right now due to a [bug in wgpu-native distribution](https://github.com/gfx-rs/wgpu-native/issues/329). I'll update these notes as soon as it gets fixed.

# SDL2-wgpu

The [`include/webgpu-demo/sdl2_wgpu.h`](https://github.com/lisyarus/webgpu-demo/blob/main/include/webgpu-demo/sdl_wgpu.h) and [`source/sdl2_wgpu.c`](https://github.com/lisyarus/webgpu-demo/blob/main/source/sdl_wgpu.c) files implement a function `WGPUSurface SDL_WGPU_CreateSurface(WGPUInstance, SDL_Window *)` which creates a WebGPU surface from an SDL2 window, and should work on Linux (X11 and Wayland), Windows and MacOS. It is mostly based on [glfw3webgpu](https://github.com/eliemichel/glfw3webgpu/blob/main/glfw3webgpu.c).

These files are almost standalone, and can be copied directly into your project, if you want to use WebGPU with SDL2. Note that the `sdl2_wgpu.c` file needs to be compiled as Objective-C for MacOS (add `-x objective-c` to compile flags for this file), and the `QuartzCore` framework needs to be linked with your application (add `-framework QuartzCore` to your linker flags).

# wgpu-native cmake find script

The [`cmake/Findwgpu-native.cmake`](https://github.com/lisyarus/webgpu-demo/blob/main/cmake/Findwgpu-native.cmake) find script is also useful on its own, and can be used in other CMake-based projects. Simply add its location to `CMAKE_MODULE_PATH`, and call `find_package(wgpu-native)`. It creates a `wgpu-native` imported library that can be simply linked to your executable via `target_link_libraries` (it sets up include directories automatically).
