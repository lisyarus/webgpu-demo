# About

A demo WebGPU renderer written partially on streams at [twitch.tv/lisyarus](twitch.tv/lisyarus), mostly for me to get familiar with WebGPU.

It uses wgpu-native WebGPU implementation, and SDL2 to create a window to render to.

# Links

* Trello board with ongoing tasks: https://trello.com/b/DlDOU1Hs/webgpu-demo
* Stream recordings: https://www.youtube.com/@lisyarus

# Building

To build this project, you need
* [CMake](https://cmake.org)
* [SDL2](https://www.libsdl.org/) (you can probably install it via your system's package manager)
* [wgpu-native](https://github.com/gfx-rs/wgpu-native)

To install wgpu-native, download [some release archive](https://github.com/gfx-rs/wgpu-native/releases) for your platform, and unpack it somewhere. ~~This project was built with the [v0.18.1.3](https://github.com/gfx-rs/wgpu-native/releases/tag/v0.18.1.3) release, and might not work with other version.~~ See **Update 2** below.

Don't forget to check out submodules:
* [glm](https://github.com/g-truc/glm) for vector & matrix maths
* [rapidjson](https://github.com/Tencent/rapidjson) for parsing glTF scenes
* [stb](https://github.com/nothings/stb) for loading images

You can do this at clone time, using `git clone <repo-url> --recurse-submodules`. Add `--shallow-submodules` to prevent loading the whole commit history of those submodules. Otherwise, you can checkout submodules at any time after cloning the repo with `git submodule update --init --recursive`.

Then, follow the usual steps for building something with CMake:
* Create a build directory
* In the build directory, run `cmake <path-to-webgpu-demo-source> -DWGPU_NATIVE_ROOT=<path-to-unpacked-wgpu-native>`
* Build the project: `cmake --build .`

Note that in case of MacOS, linking dynamic wgpu-native library (`libwgpu_native.dylib`) doesn't fully work right now due to a [bug](https://github.com/gfx-rs/wgpu-native/issues/329). The static version (`libwgpu_native.a`) works, though, so you can simply delete the dynamic library so that CMake uses the static one instead.

**Update**: this issue is fixed in the [v0.18.1.4](https://github.com/gfx-rs/wgpu-native/releases/tag/v0.18.1.4) release.

**Update 2**: filtering floating-point textures and submitting queue commands from a different thread will work reliably only in wgpu version v0.19 (scheduled to release on 14 Jan 2024), but I'm already relying on these features, so to run the project you need one of the latest trunk builds, [like this one](https://github.com/gfx-rs/wgpu-native/actions/runs/7192422937). Download the `dist` artifact from the bottom of this page, it will contain the builds for all systems and architectures.

# SDL2-wgpu

The [`include/webgpu-demo/sdl2_wgpu.h`](https://github.com/lisyarus/webgpu-demo/blob/main/include/webgpu-demo/sdl_wgpu.h) and [`source/sdl2_wgpu.c`](https://github.com/lisyarus/webgpu-demo/blob/main/source/sdl_wgpu.c) files implement a function `WGPUSurface SDL_WGPU_CreateSurface(WGPUInstance, SDL_Window *)` which creates a WebGPU surface from an SDL2 window, and should work on Linux (X11 and Wayland), Windows and MacOS. It is mostly based on [glfw3webgpu](https://github.com/eliemichel/glfw3webgpu/blob/main/glfw3webgpu.c).

These files are almost standalone, and can be copied directly into your project, if you want to use WebGPU with SDL2. Note that the `sdl2_wgpu.c` file needs to be compiled as Objective-C for MacOS (add `-x objective-c` to compile flags for this file), and the `QuartzCore` framework needs to be linked with your application (add `-framework QuartzCore` to your linker flags).

# wgpu-native cmake find script

The [`cmake/Findwgpu-native.cmake`](https://github.com/lisyarus/webgpu-demo/blob/main/cmake/Findwgpu-native.cmake) find script is also useful on its own, and can be used in other CMake-based projects. Simply add its location to `CMAKE_MODULE_PATH`, and call `find_package(wgpu-native)`. It creates a `wgpu-native` imported library that can be simply linked to your executable via `target_link_libraries` (it sets up include directories automatically).
