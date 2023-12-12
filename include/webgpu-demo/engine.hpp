#pragma once

#include <webgpu-demo/camera.hpp>
#include <webgpu-demo/box.hpp>

#include <webgpu.h>
#include <glm/glm.hpp>

#include <memory>
#include <filesystem>
#include <vector>
#include <limits>

struct RenderObject;

using RenderObjectPtr = std::shared_ptr<RenderObject>;

struct Engine
{
    Engine(WGPUDevice device, WGPUQueue queue);
    ~Engine();

    std::vector<RenderObjectPtr> loadGLTF(std::filesystem::path const & assetPath);

    void setEnvMap(std::filesystem::path const & hdrImagePath);

    Box bbox(std::vector<RenderObjectPtr> const & objects) const;

    struct LightSettings
    {
        glm::vec3 skyColor;
        glm::vec3 ambientLight;
        glm::vec3 sunDirection;
        glm::vec3 sunIntensity;
    };

    void render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, LightSettings const & lightSettings);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
