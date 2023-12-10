#include <webgpu.h>

#include <webgpu-demo/sdl_wgpu.h>
#include <webgpu-demo/application.hpp>
#include <webgpu-demo/camera.hpp>
#include <webgpu-demo/gltf_loader.hpp>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <unordered_set>

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 texcoord;
};

struct GPUMaterial
{
    glm::vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float padding1[2];
    glm::vec3 emissiveFactor;
    float padding2[1];
};

static const char shaderCode[] =
R"(

struct Material {
    baseColorFactor : vec4f,
    metallicFactor : f32,
    roughnessFactor : f32,
    emissiveFactor : vec3f,
}

@group(0) @binding(0) var<uniform> viewProjection: mat4x4f;

@group(1) @binding(0) var<uniform> model: mat4x4f;

@group(2) @binding(0) var<uniform> material : Material;
@group(2) @binding(1) var textureSampler: sampler;
@group(2) @binding(2) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(3) var normalTexture: texture_2d<f32>;

struct VertexInput {
    @builtin(vertex_index) index : u32,
    @location(0) position : vec3f,
    @location(1) normal : vec3f,
    @location(2) tangent: vec4f,
    @location(3) texcoord : vec2f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) normal : vec3f,
    @location(1) tangent : vec4f,
    @location(2) texcoord : vec2f,
}

fn asMat3x3(m : mat4x4f) -> mat3x3f {
    return mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
}

@vertex
fn vertexMain(in : VertexInput) -> VertexOutput {
    let position : vec4f = viewProjection * model * vec4f(in.position, 1.0);
    let normal : vec3f = normalize(asMat3x3(model) * in.normal);
    let tangent : vec4f = vec4f(normalize(asMat3x3(model) * in.tangent.xyz), in.tangent.w);
    return VertexOutput(position, normal, tangent, in.texcoord);
}

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4f {
    let baseColorSample = textureSample(baseColorTexture, textureSampler, in.texcoord);
    if (baseColorSample.a < 0.5) {
        discard;
    }

    var tbn = mat3x3f();
    tbn[2] = normalize(in.normal);
    tbn[0] = normalize(in.tangent.xyz - tbn[2] * dot(in.tangent.xyz, tbn[2]));
    tbn[1] = cross(tbn[2], tbn[0]) * in.tangent.w;

    let normal = tbn * normalize(2.0 * textureSample(normalTexture, textureSampler, in.texcoord).rgb - vec3(1.0));

    let lightDirection = normalize(vec3f(1.0, 2.0, 3.0));

    let lightness = 0.5 + 0.5 * dot(lightDirection, normal);

    return vec4f(baseColorSample.rgb * lightness, 1.0);
}

)";

struct Material
{
    glm::vec4 baseColorFactor;
    WGPUTextureView baseColorTextureView;

    float metallicFactor;
    float roughnessFactor;
    WGPUTextureView metallicRoughnessTextureView;

    WGPUTextureView normalTextureView;

    WGPUTextureView occlusionTextureView;

    glm::vec3 emissiveFactor;
    WGPUTextureView emissiveTextureView;
};

struct RenderObject
{
    std::uint32_t vertexByteOffset;
    std::uint32_t vertexByteLength;
    std::uint32_t vertexCount;

    std::uint32_t indexByteOffset;
    std::uint32_t indexByteLength;
    std::uint32_t indexCount;

    WGPUIndexFormat indexFormat;

    glm::mat4 modelMatrix;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;

    Material material;

    WGPUBindGroup materialBindGroup;
};

template <typename T>
struct AccessorIterator
{
    AccessorIterator(char const * ptr, std::optional<std::uint32_t> stride)
        : ptr_(ptr)
        , stride_(stride.value_or(sizeof(T)))
    {}

    T const & operator * () const
    {
        return *reinterpret_cast<T const *>(ptr_);
    }

    AccessorIterator & operator ++ ()
    {
        ptr_ += stride_;
        return *this;
    }

    AccessorIterator operator ++ (int)
    {
        auto copy = *this;
        operator++();
        return copy;
    }

private:
    char const * ptr_;
    std::size_t stride_;
};

int main()
{
    std::filesystem::path const assetPath = PROJECT_ROOT "/Sponza/Sponza.gltf";
    glTF::Asset asset = glTF::load(assetPath);

    Application application;

    std::vector<WGPUTexture> textures;

    for (auto const & textureIn : asset.textures)
    {
        auto & texture = textures.emplace_back();

        if (!textureIn.source) continue;

        auto imageInfo = glTF::loadImage(assetPath, asset.images[*textureIn.source].uri);

        WGPUTextureDescriptor textureDescriptor;
        textureDescriptor.nextInChain = nullptr;
        textureDescriptor.label = nullptr;
        textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
        textureDescriptor.dimension = WGPUTextureDimension_2D;
        textureDescriptor.size = {(std::uint32_t)imageInfo.width, (std::uint32_t)imageInfo.height, 1};

        std::optional<WGPUTextureFormat> sRGBViewFormat;

        if (imageInfo.channels == 1)
            textureDescriptor.format = WGPUTextureFormat_R8Unorm;
        else if (imageInfo.channels == 2)
            textureDescriptor.format = WGPUTextureFormat_RG8Unorm;
        else if (imageInfo.channels == 4)
        {
            textureDescriptor.format = WGPUTextureFormat_RGBA8Unorm;
            sRGBViewFormat = WGPUTextureFormat_RGBA8UnormSrgb;
        }

        textureDescriptor.mipLevelCount = 1;
        textureDescriptor.sampleCount = 1;
        textureDescriptor.viewFormatCount = sRGBViewFormat ? 1 : 0;
        textureDescriptor.viewFormats = sRGBViewFormat ? &(*sRGBViewFormat) : nullptr;

        texture = wgpuDeviceCreateTexture(application.device(), &textureDescriptor);

        WGPUImageCopyTexture imageCopyTexture;
        imageCopyTexture.nextInChain = nullptr;
        imageCopyTexture.texture = texture;
        imageCopyTexture.mipLevel = 0;
        imageCopyTexture.origin = {0, 0, 0};
        imageCopyTexture.aspect = WGPUTextureAspect_All;

        WGPUTextureDataLayout textureDataLayout;
        textureDataLayout.nextInChain = nullptr;
        textureDataLayout.offset = 0;
        textureDataLayout.bytesPerRow = imageInfo.width * imageInfo.channels;
        textureDataLayout.rowsPerImage = imageInfo.height;

        WGPUExtent3D writeSize;
        writeSize.width = imageInfo.width;
        writeSize.height = imageInfo.height;
        writeSize.depthOrArrayLayers = 1;

        wgpuQueueWriteTexture(application.queue(), &imageCopyTexture, imageInfo.data.get(), imageInfo.width * imageInfo.height * imageInfo.channels, &textureDataLayout, &writeSize);
    }

    auto createTextureView = [&](std::uint32_t textureId, bool sRGB)
    {
        WGPUTextureFormat format = wgpuTextureGetFormat(textures[textureId]);

        if (format == WGPUTextureFormat_RGBA8Unorm && sRGB)
            format = WGPUTextureFormat_RGBA8UnormSrgb;

        WGPUTextureViewDescriptor textureViewDescriptor;
        textureViewDescriptor.nextInChain = nullptr;
        textureViewDescriptor.label = nullptr;
        textureViewDescriptor.format = format;
        textureViewDescriptor.dimension = WGPUTextureViewDimension_2D;
        textureViewDescriptor.baseMipLevel = 0;
        textureViewDescriptor.mipLevelCount = 1;
        textureViewDescriptor.baseArrayLayer = 0;
        textureViewDescriptor.arrayLayerCount = 1;
        textureViewDescriptor.aspect = WGPUTextureAspect_All;

        return wgpuTextureCreateView(textures[textureId], &textureViewDescriptor);
    };

    WGPUShaderModuleWGSLDescriptor shaderModuleWGSLDescriptor;
    shaderModuleWGSLDescriptor.chain.next = nullptr;
    shaderModuleWGSLDescriptor.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    shaderModuleWGSLDescriptor.code = shaderCode;

    WGPUShaderModuleDescriptor shaderModuleDescriptor;
    shaderModuleDescriptor.nextInChain = &shaderModuleWGSLDescriptor.chain;
    shaderModuleDescriptor.label = nullptr;
    shaderModuleDescriptor.hintCount = 0;
    shaderModuleDescriptor.hints = nullptr;

    WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(application.device(), &shaderModuleDescriptor);

    WGPUBindGroupLayoutEntry cameraBindGroupLayoutEntry[1];
    cameraBindGroupLayoutEntry[0].nextInChain = nullptr;
    cameraBindGroupLayoutEntry[0].binding = 0;
    cameraBindGroupLayoutEntry[0].visibility = WGPUShaderStage_Vertex;
    cameraBindGroupLayoutEntry[0].buffer.nextInChain = nullptr;
    cameraBindGroupLayoutEntry[0].buffer.type = WGPUBufferBindingType_Uniform;
    cameraBindGroupLayoutEntry[0].buffer.hasDynamicOffset = false;
    cameraBindGroupLayoutEntry[0].buffer.minBindingSize = 64;
    cameraBindGroupLayoutEntry[0].sampler.nextInChain = nullptr;
    cameraBindGroupLayoutEntry[0].sampler.type = WGPUSamplerBindingType_Undefined;
    cameraBindGroupLayoutEntry[0].texture.nextInChain = nullptr;
    cameraBindGroupLayoutEntry[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    cameraBindGroupLayoutEntry[0].texture.multisampled = false;
    cameraBindGroupLayoutEntry[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    cameraBindGroupLayoutEntry[0].storageTexture.nextInChain = nullptr;
    cameraBindGroupLayoutEntry[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    cameraBindGroupLayoutEntry[0].storageTexture.format = WGPUTextureFormat_Undefined;
    cameraBindGroupLayoutEntry[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor cameraBindGroupLayoutDescriptor;
    cameraBindGroupLayoutDescriptor.nextInChain = nullptr;
    cameraBindGroupLayoutDescriptor.label = nullptr;
    cameraBindGroupLayoutDescriptor.entryCount = 1;
    cameraBindGroupLayoutDescriptor.entries = cameraBindGroupLayoutEntry;

    WGPUBindGroupLayoutEntry modelBindGroupLayoutEntry[1];
    modelBindGroupLayoutEntry[0].nextInChain = nullptr;
    modelBindGroupLayoutEntry[0].binding = 0;
    modelBindGroupLayoutEntry[0].visibility = WGPUShaderStage_Vertex;
    modelBindGroupLayoutEntry[0].buffer.nextInChain = nullptr;
    modelBindGroupLayoutEntry[0].buffer.type = WGPUBufferBindingType_Uniform;
    modelBindGroupLayoutEntry[0].buffer.hasDynamicOffset = false;
    modelBindGroupLayoutEntry[0].buffer.minBindingSize = 64;
    modelBindGroupLayoutEntry[0].sampler.nextInChain = nullptr;
    modelBindGroupLayoutEntry[0].sampler.type = WGPUSamplerBindingType_Undefined;
    modelBindGroupLayoutEntry[0].texture.nextInChain = nullptr;
    modelBindGroupLayoutEntry[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    modelBindGroupLayoutEntry[0].texture.multisampled = false;
    modelBindGroupLayoutEntry[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    modelBindGroupLayoutEntry[0].storageTexture.nextInChain = nullptr;
    modelBindGroupLayoutEntry[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    modelBindGroupLayoutEntry[0].storageTexture.format = WGPUTextureFormat_Undefined;
    modelBindGroupLayoutEntry[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor modelBindGroupLayoutDescriptor;
    modelBindGroupLayoutDescriptor.nextInChain = nullptr;
    modelBindGroupLayoutDescriptor.label = nullptr;
    modelBindGroupLayoutDescriptor.entryCount = 1;
    modelBindGroupLayoutDescriptor.entries = modelBindGroupLayoutEntry;

    WGPUBindGroupLayoutEntry materialBindGroupLayoutEntries[4];

    materialBindGroupLayoutEntries[0].nextInChain = nullptr;
    materialBindGroupLayoutEntries[0].binding = 0;
    materialBindGroupLayoutEntries[0].visibility = WGPUShaderStage_Fragment;
    materialBindGroupLayoutEntries[0].buffer.nextInChain = nullptr;
    materialBindGroupLayoutEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    materialBindGroupLayoutEntries[0].buffer.hasDynamicOffset = false;
    materialBindGroupLayoutEntries[0].buffer.minBindingSize = sizeof(GPUMaterial);
    materialBindGroupLayoutEntries[0].sampler.nextInChain = nullptr;
    materialBindGroupLayoutEntries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    materialBindGroupLayoutEntries[0].texture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    materialBindGroupLayoutEntries[0].texture.multisampled = false;
    materialBindGroupLayoutEntries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    materialBindGroupLayoutEntries[0].storageTexture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    materialBindGroupLayoutEntries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    materialBindGroupLayoutEntries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    materialBindGroupLayoutEntries[1].nextInChain = nullptr;
    materialBindGroupLayoutEntries[1].binding = 1;
    materialBindGroupLayoutEntries[1].visibility = WGPUShaderStage_Fragment;
    materialBindGroupLayoutEntries[1].buffer.nextInChain = nullptr;
    materialBindGroupLayoutEntries[1].buffer.type = WGPUBufferBindingType_Undefined;
    materialBindGroupLayoutEntries[1].buffer.hasDynamicOffset = false;
    materialBindGroupLayoutEntries[1].buffer.minBindingSize = 0;
    materialBindGroupLayoutEntries[1].sampler.nextInChain = nullptr;
    materialBindGroupLayoutEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;
    materialBindGroupLayoutEntries[1].texture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[1].texture.sampleType = WGPUTextureSampleType_Undefined;
    materialBindGroupLayoutEntries[1].texture.multisampled = false;
    materialBindGroupLayoutEntries[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    materialBindGroupLayoutEntries[1].storageTexture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    materialBindGroupLayoutEntries[1].storageTexture.format = WGPUTextureFormat_Undefined;
    materialBindGroupLayoutEntries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    materialBindGroupLayoutEntries[2].nextInChain = nullptr;
    materialBindGroupLayoutEntries[2].binding = 2;
    materialBindGroupLayoutEntries[2].visibility = WGPUShaderStage_Fragment;
    materialBindGroupLayoutEntries[2].buffer.nextInChain = nullptr;
    materialBindGroupLayoutEntries[2].buffer.type = WGPUBufferBindingType_Undefined;
    materialBindGroupLayoutEntries[2].buffer.hasDynamicOffset = false;
    materialBindGroupLayoutEntries[2].buffer.minBindingSize = 0;
    materialBindGroupLayoutEntries[2].sampler.nextInChain = nullptr;
    materialBindGroupLayoutEntries[2].sampler.type = WGPUSamplerBindingType_Undefined;
    materialBindGroupLayoutEntries[2].texture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[2].texture.sampleType = WGPUTextureSampleType_Float;
    materialBindGroupLayoutEntries[2].texture.multisampled = false;
    materialBindGroupLayoutEntries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
    materialBindGroupLayoutEntries[2].storageTexture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    materialBindGroupLayoutEntries[2].storageTexture.format = WGPUTextureFormat_Undefined;
    materialBindGroupLayoutEntries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    materialBindGroupLayoutEntries[3].nextInChain = nullptr;
    materialBindGroupLayoutEntries[3].binding = 3;
    materialBindGroupLayoutEntries[3].visibility = WGPUShaderStage_Fragment;
    materialBindGroupLayoutEntries[3].buffer.nextInChain = nullptr;
    materialBindGroupLayoutEntries[3].buffer.type = WGPUBufferBindingType_Undefined;
    materialBindGroupLayoutEntries[3].buffer.hasDynamicOffset = false;
    materialBindGroupLayoutEntries[3].buffer.minBindingSize = 0;
    materialBindGroupLayoutEntries[3].sampler.nextInChain = nullptr;
    materialBindGroupLayoutEntries[3].sampler.type = WGPUSamplerBindingType_Undefined;
    materialBindGroupLayoutEntries[3].texture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[3].texture.sampleType = WGPUTextureSampleType_Float;
    materialBindGroupLayoutEntries[3].texture.multisampled = false;
    materialBindGroupLayoutEntries[3].texture.viewDimension = WGPUTextureViewDimension_2D;
    materialBindGroupLayoutEntries[3].storageTexture.nextInChain = nullptr;
    materialBindGroupLayoutEntries[3].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    materialBindGroupLayoutEntries[3].storageTexture.format = WGPUTextureFormat_Undefined;
    materialBindGroupLayoutEntries[3].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor materialBindGroupLayoutDescriptor;
    materialBindGroupLayoutDescriptor.nextInChain = nullptr;
    materialBindGroupLayoutDescriptor.label = nullptr;
    materialBindGroupLayoutDescriptor.entryCount = 4;
    materialBindGroupLayoutDescriptor.entries = materialBindGroupLayoutEntries;

    WGPUBindGroupLayout bindGroupLayouts[3];
    bindGroupLayouts[0] = wgpuDeviceCreateBindGroupLayout(application.device(), &cameraBindGroupLayoutDescriptor);
    bindGroupLayouts[1] = wgpuDeviceCreateBindGroupLayout(application.device(), &modelBindGroupLayoutDescriptor);
    bindGroupLayouts[2] = wgpuDeviceCreateBindGroupLayout(application.device(), &materialBindGroupLayoutDescriptor);

    WGPUPipelineLayoutDescriptor pipelineLayoutDescriptor;
    pipelineLayoutDescriptor.nextInChain = nullptr;
    pipelineLayoutDescriptor.label = nullptr;
    pipelineLayoutDescriptor.bindGroupLayoutCount = 3;
    pipelineLayoutDescriptor.bindGroupLayouts = bindGroupLayouts;

    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(application.device(), &pipelineLayoutDescriptor);

    WGPUColorTargetState colorTargetState;
    colorTargetState.nextInChain = nullptr;
    colorTargetState.format = application.surfaceFormat();
    colorTargetState.blend = nullptr;
    colorTargetState.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState;
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "fragmentMain";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    WGPUVertexAttribute attributes[4];
    attributes[0].format = WGPUVertexFormat_Float32x3;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Float32x3;
    attributes[1].offset = 12;
    attributes[1].shaderLocation = 1;
    attributes[2].format = WGPUVertexFormat_Float32x4;
    attributes[2].offset = 24;
    attributes[2].shaderLocation = 2;
    attributes[3].format = WGPUVertexFormat_Float32x2;
    attributes[3].offset = 40;
    attributes[3].shaderLocation = 3;

    WGPUVertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.arrayStride = sizeof(Vertex);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexBufferLayout.attributeCount = 4;
    vertexBufferLayout.attributes = attributes;

    WGPUDepthStencilState depthStencilState;
    depthStencilState.nextInChain = nullptr;
    depthStencilState.format = WGPUTextureFormat_Depth24Plus;
    depthStencilState.depthWriteEnabled = true;
    depthStencilState.depthCompare = WGPUCompareFunction_Less;
    depthStencilState.stencilFront.compare = WGPUCompareFunction_Always;
    depthStencilState.stencilFront.failOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilFront.passOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.compare = WGPUCompareFunction_Always;
    depthStencilState.stencilBack.failOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.passOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilReadMask = 0;
    depthStencilState.stencilWriteMask = 0;
    depthStencilState.depthBias = 0;
    depthStencilState.depthBiasSlopeScale = 0.f;
    depthStencilState.depthBiasClamp = 0.f;

    WGPURenderPipelineDescriptor renderPipelineDescriptor;
    renderPipelineDescriptor.nextInChain = nullptr;
    renderPipelineDescriptor.label = nullptr;
    renderPipelineDescriptor.layout = pipelineLayout;
    renderPipelineDescriptor.nextInChain = nullptr;
    renderPipelineDescriptor.vertex.module = shaderModule;
    renderPipelineDescriptor.vertex.entryPoint = "vertexMain";
    renderPipelineDescriptor.vertex.constantCount = 0;
    renderPipelineDescriptor.vertex.constants = nullptr;
    renderPipelineDescriptor.vertex.bufferCount = 1;
    renderPipelineDescriptor.vertex.buffers = &vertexBufferLayout;
    renderPipelineDescriptor.primitive.nextInChain = nullptr;
    renderPipelineDescriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    renderPipelineDescriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    renderPipelineDescriptor.primitive.frontFace = WGPUFrontFace_CCW;
    renderPipelineDescriptor.primitive.cullMode = WGPUCullMode_None;
    renderPipelineDescriptor.depthStencil = &depthStencilState;
    renderPipelineDescriptor.multisample.nextInChain = nullptr;
    renderPipelineDescriptor.multisample.count = 1;
    renderPipelineDescriptor.multisample.mask = -1;
    renderPipelineDescriptor.multisample.alphaToCoverageEnabled = false;
    renderPipelineDescriptor.fragment = &fragmentState;

    WGPURenderPipeline renderPipeline = wgpuDeviceCreateRenderPipeline(application.device(), &renderPipelineDescriptor);

    std::vector<RenderObject> renderObjects;
    WGPUBuffer vertexBuffer;
    WGPUBuffer indexBuffer;

    glm::vec3 sceneBboxMin = glm::vec3( std::numeric_limits<float>::infinity());
    glm::vec3 sceneBboxMax = glm::vec3(-std::numeric_limits<float>::infinity());

    {
        if (asset.buffers.size() != 1)
            throw std::runtime_error("Only 1 binary buffer is supported");

        std::vector<char> assetBufferData = glTF::loadBuffer(assetPath, asset.buffers[0].uri);

        std::vector<Vertex> vertices;
        std::vector<std::uint16_t> indices;

        for (auto const & node : asset.nodes)
        {
            if (!node.mesh) continue;

            glm::mat4 nodeModelMatrix =
                glm::scale(node.scale) *
                glm::toMat4(node.rotation) *
                glm::translate(node.translation);

            auto const & mesh = asset.meshes[*node.mesh];

            for (auto const & primitive : mesh.primitives)
            {
                if (!primitive.attributes.position) continue;
                if (!primitive.attributes.normal) continue;
                if (!primitive.attributes.tangent) continue;
                if (!primitive.attributes.texcoord) continue;
                if (!primitive.indices) continue;

                if (primitive.mode != glTF::Primitive::Mode::Triangles) continue;

                if (!primitive.material) continue;

                auto const & positionAccessor = asset.accessors[*primitive.attributes.position];
                auto const &   normalAccessor = asset.accessors[*primitive.attributes.normal];
                auto const &  tangentAccessor = asset.accessors[*primitive.attributes.tangent];
                auto const & texcoordAccessor = asset.accessors[*primitive.attributes.texcoord];
                auto const &    indexAccessor = asset.accessors[*primitive.indices];

                if (positionAccessor.componentType != glTF::Accessor::ComponentType::Float) continue;
                if (  normalAccessor.componentType != glTF::Accessor::ComponentType::Float) continue;
                if ( tangentAccessor.componentType != glTF::Accessor::ComponentType::Float) continue;
                if (texcoordAccessor.componentType != glTF::Accessor::ComponentType::Float) continue;
                if (   indexAccessor.componentType != glTF::Accessor::ComponentType::UnsignedShort) continue;

                if (positionAccessor.type != glTF::Accessor::Type::Vec3) continue;
                if (  normalAccessor.type != glTF::Accessor::Type::Vec3) continue;
                if ( tangentAccessor.type != glTF::Accessor::Type::Vec4) continue;
                if (texcoordAccessor.type != glTF::Accessor::Type::Vec2) continue;
                if (   indexAccessor.type != glTF::Accessor::Type::Scalar) continue;

                auto const & materialIn = asset.materials[*primitive.material];

                auto & renderObject = renderObjects.emplace_back();

                renderObject.vertexByteOffset = vertices.size() * sizeof(vertices[0]);
                renderObject.vertexByteLength = positionAccessor.count * sizeof(vertices[0]);
                renderObject.vertexCount = positionAccessor.count;
                renderObject.indexByteOffset = indices.size() * sizeof(indices[0]);
                renderObject.indexByteLength = indexAccessor.count * sizeof(indices[0]);
                renderObject.indexCount = indexAccessor.count;
                renderObject.indexFormat = WGPUIndexFormat_Uint16;
                renderObject.modelMatrix = nodeModelMatrix;
                renderObject.bboxMin = glm::vec3( std::numeric_limits<float>::infinity());
                renderObject.bboxMax = glm::vec3(-std::numeric_limits<float>::infinity());

                renderObject.material.baseColorFactor = materialIn.baseColorFactor;
                renderObject.material.metallicFactor = materialIn.metallicFactor;
                renderObject.material.roughnessFactor = materialIn.roughnessFactor;
                renderObject.material.emissiveFactor = materialIn.emissiveFactor;

                if (materialIn.baseColorTexture)
                    renderObject.material.baseColorTextureView = createTextureView(*materialIn.baseColorTexture, true);

                if (materialIn.metallicRoughnessTexture)
                    renderObject.material.metallicRoughnessTextureView = createTextureView(*materialIn.metallicRoughnessTexture, false);

                if (materialIn.normalTexture)
                    renderObject.material.normalTextureView = createTextureView(*materialIn.normalTexture, false);

                if (materialIn.emissiveTexture)
                    renderObject.material.emissiveTextureView = createTextureView(*materialIn.emissiveTexture, false);

                auto const & positionBufferView = asset.bufferViews[positionAccessor.bufferView];
                auto const &   normalBufferView = asset.bufferViews[  normalAccessor.bufferView];
                auto const &  tangentBufferView = asset.bufferViews[ tangentAccessor.bufferView];
                auto const & texcoordBufferView = asset.bufferViews[texcoordAccessor.bufferView];
                auto const &    indexBufferView = asset.bufferViews[   indexAccessor.bufferView];

                auto positionIterator = AccessorIterator<glm::vec3>(assetBufferData.data() + positionBufferView.byteOffset + positionAccessor.byteOffset, positionBufferView.byteStride);
                auto normalIterator = AccessorIterator<glm::vec3>(assetBufferData.data() + normalBufferView.byteOffset + normalAccessor.byteOffset, normalBufferView.byteStride);
                auto tangentIterator = AccessorIterator<glm::vec4>(assetBufferData.data() + tangentBufferView.byteOffset + tangentAccessor.byteOffset, tangentBufferView.byteStride);
                auto texcoordIterator = AccessorIterator<glm::vec2>(assetBufferData.data() + texcoordBufferView.byteOffset + texcoordAccessor.byteOffset, texcoordBufferView.byteStride);

                for (int i = 0; i < positionAccessor.count; ++i)
                {
                    vertices.push_back({
                        *positionIterator++,
                        *  normalIterator++,
                        * tangentIterator++,
                        *texcoordIterator++,
                    });

                    auto transformedVertex = glm::vec3((renderObject.modelMatrix * glm::vec4(vertices.back().position, 1.f)));

                    renderObject.bboxMin = glm::min(renderObject.bboxMin, transformedVertex); renderObject.bboxMax = glm::max(renderObject.bboxMax, transformedVertex);
                }

                sceneBboxMin = glm::min(sceneBboxMin, renderObject.bboxMin);
                sceneBboxMax = glm::max(sceneBboxMax, renderObject.bboxMax);

                auto indexIterator = AccessorIterator<std::uint16_t>(assetBufferData.data() + indexBufferView.byteOffset + indexAccessor.byteOffset, indexBufferView.byteStride);
                for (int i = 0; i < indexAccessor.count; ++i)
                    indices.emplace_back(*indexIterator++);
            }
        }

        // Respect COPY_BUFFER_ALIGNMENT
        if ((indices.size() % 2) != 0)
            indices.push_back(0);

        WGPUBufferDescriptor vertexBufferDescriptor;
        vertexBufferDescriptor.nextInChain = nullptr;
        vertexBufferDescriptor.label = nullptr;
        vertexBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
        vertexBufferDescriptor.size = vertices.size() * sizeof(vertices[0]);
        vertexBufferDescriptor.mappedAtCreation = false;

        vertexBuffer = wgpuDeviceCreateBuffer(application.device(), &vertexBufferDescriptor);

        wgpuQueueWriteBuffer(application.queue(), vertexBuffer, 0, vertices.data(), vertices.size() * sizeof(vertices[0]));

        WGPUBufferDescriptor indexBufferDescriptor;
        indexBufferDescriptor.nextInChain = nullptr;
        indexBufferDescriptor.label = nullptr;
        indexBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index;
        indexBufferDescriptor.size = indices.size() * sizeof(indices[0]);
        indexBufferDescriptor.mappedAtCreation = false;

        indexBuffer = wgpuDeviceCreateBuffer(application.device(), &indexBufferDescriptor);

        wgpuQueueWriteBuffer(application.queue(), indexBuffer, 0, indices.data(), indices.size() * sizeof(indices[0]));
    }

    WGPUBufferDescriptor cameraUniformBufferDescriptor;
    cameraUniformBufferDescriptor.nextInChain = nullptr;
    cameraUniformBufferDescriptor.label = nullptr;
    cameraUniformBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    cameraUniformBufferDescriptor.size = 64;
    cameraUniformBufferDescriptor.mappedAtCreation = false;

    WGPUBuffer cameraUniformBuffer = wgpuDeviceCreateBuffer(application.device(), &cameraUniformBufferDescriptor);

    WGPUBufferDescriptor modelUniformBufferDescriptor;
    modelUniformBufferDescriptor.nextInChain = nullptr;
    modelUniformBufferDescriptor.label = nullptr;
    modelUniformBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    modelUniformBufferDescriptor.size = 64;
    modelUniformBufferDescriptor.mappedAtCreation = false;

    WGPUBuffer modelUniformBuffer = wgpuDeviceCreateBuffer(application.device(), &modelUniformBufferDescriptor);

    WGPUBufferDescriptor materialUniformBufferDescriptor;
    materialUniformBufferDescriptor.nextInChain = nullptr;
    materialUniformBufferDescriptor.label = nullptr;
    materialUniformBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    materialUniformBufferDescriptor.size = sizeof(GPUMaterial);
    materialUniformBufferDescriptor.mappedAtCreation = false;

    WGPUBuffer materialUniformBuffer = wgpuDeviceCreateBuffer(application.device(), &materialUniformBufferDescriptor);

    WGPUBindGroupEntry cameraBindGroupEntry;
    cameraBindGroupEntry.nextInChain = nullptr;
    cameraBindGroupEntry.binding = 0;
    cameraBindGroupEntry.buffer = cameraUniformBuffer;
    cameraBindGroupEntry.offset = 0;
    cameraBindGroupEntry.size = 64;
    cameraBindGroupEntry.sampler = nullptr;
    cameraBindGroupEntry.textureView = nullptr;

    WGPUBindGroupDescriptor cameraBindGroupDescriptor;
    cameraBindGroupDescriptor.nextInChain = nullptr;
    cameraBindGroupDescriptor.label = nullptr;
    cameraBindGroupDescriptor.layout = bindGroupLayouts[0];
    cameraBindGroupDescriptor.entryCount = 1;
    cameraBindGroupDescriptor.entries = &cameraBindGroupEntry;

    WGPUBindGroupEntry modelBindGroupEntry;
    modelBindGroupEntry.nextInChain = nullptr;
    modelBindGroupEntry.binding = 0;
    modelBindGroupEntry.buffer = modelUniformBuffer;
    modelBindGroupEntry.offset = 0;
    modelBindGroupEntry.size = 64;
    modelBindGroupEntry.sampler = nullptr;
    modelBindGroupEntry.textureView = nullptr;

    WGPUBindGroupDescriptor modelBindGroupDescriptor;
    modelBindGroupDescriptor.nextInChain = nullptr;
    modelBindGroupDescriptor.label = nullptr;
    modelBindGroupDescriptor.layout = bindGroupLayouts[1];
    modelBindGroupDescriptor.entryCount = 1;
    modelBindGroupDescriptor.entries = &modelBindGroupEntry;

    auto cameraBindGroup = wgpuDeviceCreateBindGroup(application.device(), &cameraBindGroupDescriptor);
    auto modelBindGroup = wgpuDeviceCreateBindGroup(application.device(), &modelBindGroupDescriptor);

    WGPUSamplerDescriptor samplerDescriptor;
    samplerDescriptor.nextInChain = nullptr;
    samplerDescriptor.label = nullptr;
    samplerDescriptor.addressModeU = WGPUAddressMode_MirrorRepeat;
    samplerDescriptor.addressModeV = WGPUAddressMode_MirrorRepeat;
    samplerDescriptor.addressModeW = WGPUAddressMode_MirrorRepeat;
    samplerDescriptor.magFilter = WGPUFilterMode_Linear;
    samplerDescriptor.minFilter = WGPUFilterMode_Linear;
    samplerDescriptor.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    samplerDescriptor.lodMinClamp = 0.f;
    samplerDescriptor.lodMaxClamp = 0.f;
    samplerDescriptor.compare = WGPUCompareFunction_Undefined;
    samplerDescriptor.maxAnisotropy = 1;

    WGPUSampler sampler = wgpuDeviceCreateSampler(application.device(), &samplerDescriptor);

    for (auto & renderObject : renderObjects)
    {
        WGPUBindGroupEntry materialBindGroupEntries[4];

        materialBindGroupEntries[0].nextInChain = nullptr;
        materialBindGroupEntries[0].binding = 0;
        materialBindGroupEntries[0].buffer = materialUniformBuffer;
        materialBindGroupEntries[0].offset = 0;
        materialBindGroupEntries[0].size = sizeof(GPUMaterial);
        materialBindGroupEntries[0].sampler = nullptr;
        materialBindGroupEntries[0].textureView = nullptr;

        materialBindGroupEntries[1].nextInChain = nullptr;
        materialBindGroupEntries[1].binding = 1;
        materialBindGroupEntries[1].buffer = nullptr;
        materialBindGroupEntries[1].offset = 0;
        materialBindGroupEntries[1].size = 0;
        materialBindGroupEntries[1].sampler = sampler;
        materialBindGroupEntries[1].textureView = nullptr;

        materialBindGroupEntries[2].nextInChain = nullptr;
        materialBindGroupEntries[2].binding = 2;
        materialBindGroupEntries[2].buffer = nullptr;
        materialBindGroupEntries[2].offset = 0;
        materialBindGroupEntries[2].size = 0;
        materialBindGroupEntries[2].sampler = nullptr;
        materialBindGroupEntries[2].textureView = renderObject.material.baseColorTextureView;

        materialBindGroupEntries[3].nextInChain = nullptr;
        materialBindGroupEntries[3].binding = 3;
        materialBindGroupEntries[3].buffer = nullptr;
        materialBindGroupEntries[3].offset = 0;
        materialBindGroupEntries[3].size = 0;
        materialBindGroupEntries[3].sampler = nullptr;
        materialBindGroupEntries[3].textureView = renderObject.material.normalTextureView;

        WGPUBindGroupDescriptor materialBindGroupDescriptor;
        materialBindGroupDescriptor.nextInChain = nullptr;
        materialBindGroupDescriptor.label = nullptr;
        materialBindGroupDescriptor.layout = bindGroupLayouts[2];
        materialBindGroupDescriptor.entryCount = 4;
        materialBindGroupDescriptor.entries = materialBindGroupEntries;

        renderObject.materialBindGroup = wgpuDeviceCreateBindGroup(application.device(), &materialBindGroupDescriptor);
    }

    WGPUTexture depthTexture = nullptr;
    WGPUTextureView depthTextureView = nullptr;

    Camera camera;
    camera.move((sceneBboxMin + sceneBboxMax) / 2.f);
    camera.setFov(glm::radians(45.f), application.width() * 1.f / application.height());
    float sceneDiagonal = glm::distance(sceneBboxMin, sceneBboxMax);
    camera.setClip(sceneDiagonal / 1000.f, sceneDiagonal);
    camera.setSpeed(sceneDiagonal / 10.f);

    std::unordered_set<SDL_Scancode> keysDown;

    int frameId = 0;
    float time = 0.f;

    auto lastFrameStart = std::chrono::high_resolution_clock::now();

    for (bool running = true; running;)
    {
        std::cout << "Frame " << frameId << std::endl;

        bool resized = false;

        while (auto event = application.poll()) switch (event->type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_WINDOWEVENT:
            switch (event->window.event)
            {
            case SDL_WINDOWEVENT_RESIZED:
                application.resize(event->window.data1, event->window.data2);
                camera.setFov(glm::radians(45.f), application.width() * 1.f / application.height());
                resized = true;
                break;
            }
            break;
        case SDL_MOUSEMOTION:
            camera.rotate(event->motion.xrel, event->motion.yrel);
            break;
        case SDL_KEYDOWN:
            keysDown.insert(event->key.keysym.scancode);
            break;
        case SDL_KEYUP:
            keysDown.erase(event->key.keysym.scancode);
            break;
        }

        auto surfaceTextureView = application.nextSwapchainView();
        if (!surfaceTextureView)
        {
            ++frameId;
            continue;
        }

        auto thisFrameStart = std::chrono::high_resolution_clock::now();
        float const dt = std::chrono::duration_cast<std::chrono::duration<float>>(thisFrameStart - lastFrameStart).count();
        time += dt;
        lastFrameStart = thisFrameStart;

        camera.update(dt, {
            .movingForward  = keysDown.contains(SDL_SCANCODE_W),
            .movingBackward = keysDown.contains(SDL_SCANCODE_S),
            .movingLeft     = keysDown.contains(SDL_SCANCODE_A),
            .movingRight    = keysDown.contains(SDL_SCANCODE_D),
            .movingFast     = keysDown.contains(SDL_SCANCODE_LSHIFT),
            .movingSlow     = keysDown.contains(SDL_SCANCODE_LCTRL),
        });

        glm::mat4 const viewProjectionMatrix = camera.viewProjectionMatrix();

        wgpuQueueWriteBuffer(application.queue(), cameraUniformBuffer, 0, &viewProjectionMatrix, 64);

        if (resized || !depthTexture)
        {
            if (depthTexture)
            {
                wgpuTextureViewRelease(depthTextureView);
                wgpuTextureRelease(depthTexture);
            }

            WGPUTextureDescriptor depthTextureDescriptor;
            depthTextureDescriptor.nextInChain = nullptr;
            depthTextureDescriptor.label = nullptr;
            depthTextureDescriptor.usage = WGPUTextureUsage_RenderAttachment;
            depthTextureDescriptor.dimension = WGPUTextureDimension_2D;
            depthTextureDescriptor.size = {(std::uint32_t)application.width(), (std::uint32_t)application.height(), 1};
            depthTextureDescriptor.format = WGPUTextureFormat_Depth24Plus;
            depthTextureDescriptor.mipLevelCount = 1;
            depthTextureDescriptor.sampleCount = 1;
            depthTextureDescriptor.viewFormatCount = 1;
            depthTextureDescriptor.viewFormats = &depthTextureDescriptor.format;

            depthTexture = wgpuDeviceCreateTexture(application.device(), &depthTextureDescriptor);

            WGPUTextureViewDescriptor depthTextureViewDescriptor;
            depthTextureViewDescriptor.nextInChain = nullptr;
            depthTextureViewDescriptor.label = nullptr;
            depthTextureViewDescriptor.format = WGPUTextureFormat_Depth24Plus;
            depthTextureViewDescriptor.dimension = WGPUTextureViewDimension_2D;
            depthTextureViewDescriptor.baseMipLevel = 0;
            depthTextureViewDescriptor.mipLevelCount = 1;
            depthTextureViewDescriptor.baseArrayLayer = 0;
            depthTextureViewDescriptor.arrayLayerCount = 1;
            depthTextureViewDescriptor.aspect = WGPUTextureAspect_DepthOnly;

            depthTextureView = wgpuTextureCreateView(depthTexture, &depthTextureViewDescriptor);
        }

        WGPUCommandEncoderDescriptor commandEncoderDescriptor;
        commandEncoderDescriptor.nextInChain = nullptr;
        commandEncoderDescriptor.label = nullptr;

        WGPUCommandEncoder commandEncoder = wgpuDeviceCreateCommandEncoder(application.device(), &commandEncoderDescriptor);

        WGPURenderPassColorAttachment renderPassColorAttachment;
        renderPassColorAttachment.nextInChain = nullptr;
        renderPassColorAttachment.view = surfaceTextureView;
        renderPassColorAttachment.resolveTarget = nullptr;
        renderPassColorAttachment.loadOp = WGPULoadOp_Clear;
        renderPassColorAttachment.storeOp = WGPUStoreOp_Store;
        renderPassColorAttachment.clearValue = {0.8, 0.9, 1.0, 1.0};

        WGPURenderPassDepthStencilAttachment renderPassDepthStencilAttachment;
        renderPassDepthStencilAttachment.view = depthTextureView;
        renderPassDepthStencilAttachment.depthLoadOp = WGPULoadOp_Clear;
        renderPassDepthStencilAttachment.depthStoreOp = WGPUStoreOp_Store;
        renderPassDepthStencilAttachment.depthClearValue = 1.f;
        renderPassDepthStencilAttachment.depthReadOnly = false;
        renderPassDepthStencilAttachment.stencilLoadOp = WGPULoadOp_Undefined;
        renderPassDepthStencilAttachment.stencilStoreOp = WGPUStoreOp_Undefined;
        renderPassDepthStencilAttachment.stencilClearValue = 0;
        renderPassDepthStencilAttachment.stencilReadOnly = true;

        WGPURenderPassDescriptor renderPassDescriptor;
        renderPassDescriptor.nextInChain = nullptr;
        renderPassDescriptor.label = nullptr;
        renderPassDescriptor.colorAttachmentCount = 1;
        renderPassDescriptor.colorAttachments = &renderPassColorAttachment;
        renderPassDescriptor.depthStencilAttachment = &renderPassDepthStencilAttachment;
        renderPassDescriptor.occlusionQuerySet = nullptr;
        renderPassDescriptor.timestampWrites = nullptr;

        WGPURenderPassEncoder renderPassEncoder = wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPassDescriptor);

        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, renderPipeline);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, cameraBindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, modelBindGroup, 0, nullptr);

        for (auto const & renderObject : renderObjects)
        {
            GPUMaterial material;
            material.baseColorFactor = renderObject.material.baseColorFactor;
            material.metallicFactor = renderObject.material.metallicFactor;
            material.roughnessFactor = renderObject.material.roughnessFactor;
            material.emissiveFactor = renderObject.material.emissiveFactor;

            wgpuQueueWriteBuffer(application.queue(), modelUniformBuffer, 0, &renderObject.modelMatrix, 64);
            wgpuQueueWriteBuffer(application.queue(), materialUniformBuffer, 0, &material, sizeof(material));

            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, renderObject.materialBindGroup, 0, nullptr);

            wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 0, vertexBuffer, renderObject.vertexByteOffset, renderObject.vertexByteLength);
            wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder, indexBuffer, renderObject.indexFormat, renderObject.indexByteOffset, renderObject.indexByteLength);
            wgpuRenderPassEncoderDrawIndexed(renderPassEncoder, renderObject.indexCount, 1, 0, 0, 0);
        }

        wgpuRenderPassEncoderEnd(renderPassEncoder);

        wgpuTextureViewRelease(surfaceTextureView);

        WGPUCommandBufferDescriptor commandBufferDescriptor;
        commandBufferDescriptor.nextInChain = nullptr;
        commandBufferDescriptor.label = nullptr;

        WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(commandEncoder, &commandBufferDescriptor);
        wgpuQueueSubmit(application.queue(), 1, &commandBuffer);

        application.present();

        ++frameId;
    }
}
