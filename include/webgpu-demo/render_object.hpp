#pragma once

#include <webgpu-demo/engine_utils.hpp>
#include <webgpu-demo/box.hpp>

#include <glm/glm.hpp>

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>
#include <optional>

struct RenderObject;

struct RenderObjectCommon
{
    WGPUBuffer vertexBuffer;
    WGPUBuffer indexBuffer;

    WGPUTexture whiteTexture;

    struct TextureInfo
    {
        std::atomic<WGPUTexture> texture = nullptr;

        std::filesystem::path assetPath;
        std::string uri;
        bool sRGB = false;

        std::vector<std::weak_ptr<RenderObject>> users;
    };

    std::vector<std::unique_ptr<TextureInfo>> textures;

    WGPUTextureView createTextureView(std::optional<std::uint32_t> textureId, bool sRGB);

    ~RenderObjectCommon();
};

struct RenderObject
{
    std::shared_ptr<RenderObjectCommon> common;

    std::uint32_t vertexByteOffset;
    std::uint32_t vertexByteLength;
    std::uint32_t vertexCount;

    std::uint32_t indexByteOffset;
    std::uint32_t indexByteLength;
    std::uint32_t indexCount;

    WGPUIndexFormat indexFormat;

    ObjectUniform uniforms;

    Box bbox;

    struct Textures
    {
        std::optional<std::uint32_t> baseColorTextureId;
        std::optional<std::uint32_t> metallicRoughnessTextureId;
        std::optional<std::uint32_t> normalTextureId;
    };

    Textures textures;

    WGPUBindGroup texturesBindGroup = nullptr;

    void createTexturesBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, WGPUSampler sampler);

    ~RenderObject();
};
