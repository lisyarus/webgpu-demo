#pragma once

#include <webgpu-demo/camera.hpp>

#include <webgpu.h>
#include <glm/glm.hpp>

#include <memory>
#include <filesystem>
#include <vector>

struct RenderObject;

using RenderObjectPtr = std::shared_ptr<RenderObject>;

struct Engine
{
    Engine(WGPUDevice device, WGPUQueue queue);
    ~Engine();

    std::vector<RenderObjectPtr> loadGLTF(std::filesystem::path const & assetPath);

    std::pair<glm::vec3, glm::vec3> bbox(std::vector<RenderObjectPtr> const & objects) const;

    void render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
