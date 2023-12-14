#include <webgpu-demo/render_object.hpp>

WGPUTextureView RenderObjectCommon::createTextureView(std::optional<std::uint32_t> textureId, bool sRGB)
{
    WGPUTexture texture = nullptr;
    if (textureId)
        texture = textures[*textureId]->texture.load();
    if (!texture)
        texture = whiteTexture;

    WGPUTextureFormat format = wgpuTextureGetFormat(texture);
    int mipLevelsCount = wgpuTextureGetMipLevelCount(texture);

    if (format == WGPUTextureFormat_RGBA8Unorm && sRGB)
        format = WGPUTextureFormat_RGBA8UnormSrgb;

    WGPUTextureViewDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.format = format;
    descriptor.dimension = WGPUTextureViewDimension_2D;
    descriptor.baseMipLevel = 0;
    descriptor.mipLevelCount = mipLevelsCount;
    descriptor.baseArrayLayer = 0;
    descriptor.arrayLayerCount = 1;
    descriptor.aspect = WGPUTextureAspect_All;

    return wgpuTextureCreateView(texture, &descriptor);
}

RenderObjectCommon::~RenderObjectCommon()
{
    for (auto const & textureInfo : textures)
        if (auto texture = textureInfo->texture.load())
            wgpuTextureRelease(texture);

    wgpuBufferRelease(indexBuffer);
    wgpuBufferRelease(vertexBuffer);
}

void RenderObject::createTexturesBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, WGPUSampler sampler)
{
    if (texturesBindGroup)
    {
        wgpuBindGroupRelease(texturesBindGroup);
        texturesBindGroup = nullptr;
    }

    WGPUBindGroupEntry entries[4];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = nullptr;
    entries[0].offset = 0;
    entries[0].size = 0;
    entries[0].sampler = sampler;
    entries[0].textureView = nullptr;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].buffer = nullptr;
    entries[1].offset = 0;
    entries[1].size = 0;
    entries[1].sampler = nullptr;
    entries[1].textureView = common->createTextureView(textures.baseColorTextureId, true);

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].buffer = nullptr;
    entries[2].offset = 0;
    entries[2].size = 0;
    entries[2].sampler = nullptr;
    entries[2].textureView = common->createTextureView(textures.normalTextureId, false);

    entries[3].nextInChain = nullptr;
    entries[3].binding = 3;
    entries[3].buffer = nullptr;
    entries[3].offset = 0;
    entries[3].size = 0;
    entries[3].sampler = nullptr;
    entries[3].textureView = common->createTextureView(textures.metallicRoughnessTextureId, false);

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = layout;
    descriptor.entryCount = 4;
    descriptor.entries = entries;

    texturesBindGroup = wgpuDeviceCreateBindGroup(device, &descriptor);

    for (auto const & entry : entries)
        if (entry.textureView)
            wgpuTextureViewRelease(entry.textureView);
}

RenderObject::~RenderObject()
{
    if (texturesBindGroup)
        wgpuBindGroupRelease(texturesBindGroup);
}
