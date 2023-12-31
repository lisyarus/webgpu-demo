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
    WGPUBuffer clothEdgesBuffer;
    WGPUBuffer clothVertexBuffer;

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

    struct BufferData
    {
        std::uint32_t byteOffset;
        std::uint32_t byteLength;
        std::uint32_t count;
    };

    BufferData vertices;
    BufferData indices;

    struct ClothData
    {
        BufferData edges;
        BufferData vertices;
    };

    std::optional<ClothData> cloth;

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
    WGPUBindGroup clothBindGroup = nullptr;

    struct Light
    {
        glm::vec3 position;
        glm::vec3 intensity;
    };

    std::optional<Light> light;

    void createTexturesBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, WGPUSampler sampler);
    void createClothBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, WGPUBuffer settingsBuffer);

    ~RenderObject();
};
