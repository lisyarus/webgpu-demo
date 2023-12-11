#pragma once

#include <webgpu-demo/camera.hpp>

#include <webgpu.h>
#include <glm/glm.hpp>

#include <memory>
#include <filesystem>
#include <vector>
#include <limits>

struct RenderObject;

using RenderObjectPtr = std::shared_ptr<RenderObject>;

struct Box
{
    glm::vec3 min{ std::numeric_limits<float>::infinity()};
    glm::vec3 max{-std::numeric_limits<float>::infinity()};

    Box & expand(glm::vec3 const & p)
    {
        min = glm::min(min, p);
        max = glm::max(max, p);
        return *this;
    }

    Box & expand(Box const & b)
    {
        min = glm::min(min, b.min);
        max = glm::max(max, b.max);
        return *this;
    }
};

struct Engine
{
    Engine(WGPUDevice device, WGPUQueue queue);
    ~Engine();

    std::vector<RenderObjectPtr> loadGLTF(std::filesystem::path const & assetPath);

    Box bbox(std::vector<RenderObjectPtr> const & objects) const;

    void render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
