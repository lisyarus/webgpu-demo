#pragma once

#ifdef __APPLE__
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif

#include <webgpu.h>

#include <optional>

struct Application
{
    Application();

    SDL_Window * window() const { return window_; }

    WGPUSurface surface() const { return surface_; }

    WGPUTextureFormat surfaceFormat() const { return surfaceFormat_; }

    WGPUDevice device() const { return device_; }

    WGPUQueue queue() const { return queue_; }

    int width() const { return width_; }
    int height() const { return height_; }

    void resize(int width, int height);

    WGPUTextureView nextSwapchainView();

    void present();

    std::optional<SDL_Event> poll();

private:
    SDL_Window * window_;
    WGPUSurface surface_;
    WGPUTextureFormat surfaceFormat_;
    WGPUDevice device_;
    WGPUQueue queue_;

    int width_ = 1024;
    int height_ = 768;
};
