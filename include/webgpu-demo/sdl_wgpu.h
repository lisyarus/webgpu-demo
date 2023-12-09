#pragma once

#include <SDL2/SDL_video.h>
#include <webgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

WGPUSurface SDL_WGPU_CreateSurface(WGPUInstance instance, SDL_Window * window);

#ifdef __cplusplus
}
#endif
