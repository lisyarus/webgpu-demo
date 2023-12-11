#include <webgpu-demo/engine.hpp>
#include <webgpu-demo/gltf_loader.hpp>
#include <webgpu-demo/gltf_iterator.hpp>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>

namespace
{

    struct Vertex
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec4 tangent;
        glm::vec2 texcoord;
    };

    struct CameraUniform
    {
        glm::mat4 viewProjection;
        glm::mat4 viewProjectionInverse;
        glm::vec3 position;
        float padding1[1];
    };

    struct MaterialUniform
    {
        glm::vec4 baseColorFactor;
        float metallicFactor;
        float roughnessFactor;
        float padding1[2];
        glm::vec3 emissiveFactor;
        float padding2[1];
    };

    static const char mainShader[] =
R"(

struct Camera {
    viewProjection : mat4x4f,
    viewProjectionInverse : mat4x4f,
    position : vec3f,
}

struct Material {
    baseColorFactor : vec4f,
    metallicFactor : f32,
    roughnessFactor : f32,
    emissiveFactor : vec3f,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform> model: mat4x4f;

@group(2) @binding(0) var<uniform> material : Material;
@group(2) @binding(1) var textureSampler: sampler;
@group(2) @binding(2) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(3) var normalTexture: texture_2d<f32>;
@group(2) @binding(4) var metallicRoughnessTexture: texture_2d<f32>;

struct VertexInput {
    @builtin(vertex_index) index : u32,
    @location(0) position : vec3f,
    @location(1) normal : vec3f,
    @location(2) tangent: vec4f,
    @location(3) texcoord : vec2f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) worldPosition : vec3f,
    @location(1) normal : vec3f,
    @location(2) tangent : vec4f,
    @location(3) texcoord : vec2f,
}

fn asMat3x3(m : mat4x4f) -> mat3x3f {
    return mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
}

@vertex
fn vertexMain(in : VertexInput) -> VertexOutput {
    let worldPosition = (model * vec4f(in.position, 1.0)).xyz;
    let position : vec4f = camera.viewProjection * vec4f(worldPosition, 1.0);
    let normal : vec3f = normalize(asMat3x3(model) * in.normal);
    let tangent : vec4f = vec4f(normalize(asMat3x3(model) * in.tangent.xyz), in.tangent.w);
    return VertexOutput(position, worldPosition, normal, tangent, in.texcoord);
}

fn Uncharted2TonemapImpl(x : vec3f) -> vec3f {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;

    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

fn ACESFilm(x : vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

fn tonemap(x : vec3f) -> vec3f {
    return Uncharted2TonemapImpl(x) / Uncharted2TonemapImpl(vec3f(3.0));
//    return ACESFilm(x);
}

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4f {
    let baseColorSample = textureSample(baseColorTexture, textureSampler, in.texcoord) * material.baseColorFactor;

    let baseColor = baseColorSample.rgb;

    var tbn = mat3x3f();
    tbn[2] = normalize(in.normal);
    tbn[0] = normalize(in.tangent.xyz - tbn[2] * dot(in.tangent.xyz, tbn[2]));
    tbn[1] = cross(tbn[2], tbn[0]) * in.tangent.w;

    let normal = tbn * normalize(2.0 * textureSample(normalTexture, textureSampler, in.texcoord).rgb - vec3(1.0));

    let materialSample = textureSample(metallicRoughnessTexture, textureSampler, in.texcoord);

    if (baseColorSample.a < 0.5) {
        discard;
    }

    let metallic = materialSample.b * material.metallicFactor;
    let roughness = materialSample.g * material.roughnessFactor;

    let ambientLight = vec3f(0.5);
    let lightDirection = normalize(vec3f(1.0, 2.0, 3.0));
    let lightIntensity = vec3f(10.0, 8.0, 6.0);

    let viewDirection = normalize(camera.position - in.worldPosition);
    let halfway = normalize(lightDirection + viewDirection);

    let lightness = max(0.0, dot(normal, lightDirection));
    let specular = lightness * metallic * pow(max(0.0, dot(viewDirection, halfway)), 1.0 / max(0.0001, roughness * roughness));

    let outColor = (ambientLight + (lightness + specular) * lightIntensity) * baseColor;

    return vec4f(tonemap(outColor), 1.0);
}

)";

    WGPUShaderModule createShaderModule(WGPUDevice device, char const * code)
    {
        WGPUShaderModuleWGSLDescriptor shaderModuleWGSLDescriptor;
        shaderModuleWGSLDescriptor.chain.next = nullptr;
        shaderModuleWGSLDescriptor.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
        shaderModuleWGSLDescriptor.code = code;

        WGPUShaderModuleDescriptor shaderModuleDescriptor;
        shaderModuleDescriptor.nextInChain = &shaderModuleWGSLDescriptor.chain;
        shaderModuleDescriptor.label = nullptr;
        shaderModuleDescriptor.hintCount = 0;
        shaderModuleDescriptor.hints = nullptr;

        return wgpuDeviceCreateShaderModule(device, &shaderModuleDescriptor);
    }

    WGPUBindGroupLayout createCameraBindGroupLayout(WGPUDevice device)
    {
        WGPUBindGroupLayoutEntry entries[1];
        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        entries[0].buffer.nextInChain = nullptr;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.hasDynamicOffset = false;
        entries[0].buffer.minBindingSize = sizeof(CameraUniform);
        entries[0].sampler.nextInChain = nullptr;
        entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
        entries[0].texture.nextInChain = nullptr;
        entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
        entries[0].texture.multisampled = false;
        entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
        entries[0].storageTexture.nextInChain = nullptr;
        entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        WGPUBindGroupLayoutDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.entryCount = 1;
        descriptor.entries = entries;

        return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    }

    WGPUBindGroupLayout createModelBindGroupLayout(WGPUDevice device)
    {
        WGPUBindGroupLayoutEntry entries[1];
        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Vertex;
        entries[0].buffer.nextInChain = nullptr;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.hasDynamicOffset = false;
        entries[0].buffer.minBindingSize = 64;
        entries[0].sampler.nextInChain = nullptr;
        entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
        entries[0].texture.nextInChain = nullptr;
        entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
        entries[0].texture.multisampled = false;
        entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
        entries[0].storageTexture.nextInChain = nullptr;
        entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        WGPUBindGroupLayoutDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.entryCount = 1;
        descriptor.entries = entries;

        return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    }

    WGPUBindGroupLayout createMaterialBindGroupLayout(WGPUDevice device)
    {
        WGPUBindGroupLayoutEntry entries[5];

        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Fragment;
        entries[0].buffer.nextInChain = nullptr;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.hasDynamicOffset = false;
        entries[0].buffer.minBindingSize = sizeof(MaterialUniform);
        entries[0].sampler.nextInChain = nullptr;
        entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
        entries[0].texture.nextInChain = nullptr;
        entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
        entries[0].texture.multisampled = false;
        entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
        entries[0].storageTexture.nextInChain = nullptr;
        entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        entries[1].nextInChain = nullptr;
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].buffer.nextInChain = nullptr;
        entries[1].buffer.type = WGPUBufferBindingType_Undefined;
        entries[1].buffer.hasDynamicOffset = false;
        entries[1].buffer.minBindingSize = 0;
        entries[1].sampler.nextInChain = nullptr;
        entries[1].sampler.type = WGPUSamplerBindingType_Filtering;
        entries[1].texture.nextInChain = nullptr;
        entries[1].texture.sampleType = WGPUTextureSampleType_Undefined;
        entries[1].texture.multisampled = false;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
        entries[1].storageTexture.nextInChain = nullptr;
        entries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[1].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        entries[2].nextInChain = nullptr;
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Fragment;
        entries[2].buffer.nextInChain = nullptr;
        entries[2].buffer.type = WGPUBufferBindingType_Undefined;
        entries[2].buffer.hasDynamicOffset = false;
        entries[2].buffer.minBindingSize = 0;
        entries[2].sampler.nextInChain = nullptr;
        entries[2].sampler.type = WGPUSamplerBindingType_Undefined;
        entries[2].texture.nextInChain = nullptr;
        entries[2].texture.sampleType = WGPUTextureSampleType_Float;
        entries[2].texture.multisampled = false;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
        entries[2].storageTexture.nextInChain = nullptr;
        entries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[2].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        entries[3].nextInChain = nullptr;
        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Fragment;
        entries[3].buffer.nextInChain = nullptr;
        entries[3].buffer.type = WGPUBufferBindingType_Undefined;
        entries[3].buffer.hasDynamicOffset = false;
        entries[3].buffer.minBindingSize = 0;
        entries[3].sampler.nextInChain = nullptr;
        entries[3].sampler.type = WGPUSamplerBindingType_Undefined;
        entries[3].texture.nextInChain = nullptr;
        entries[3].texture.sampleType = WGPUTextureSampleType_Float;
        entries[3].texture.multisampled = false;
        entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;
        entries[3].storageTexture.nextInChain = nullptr;
        entries[3].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[3].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        entries[4].nextInChain = nullptr;
        entries[4].binding = 4;
        entries[4].visibility = WGPUShaderStage_Fragment;
        entries[4].buffer.nextInChain = nullptr;
        entries[4].buffer.type = WGPUBufferBindingType_Undefined;
        entries[4].buffer.hasDynamicOffset = false;
        entries[4].buffer.minBindingSize = 0;
        entries[4].sampler.nextInChain = nullptr;
        entries[4].sampler.type = WGPUSamplerBindingType_Undefined;
        entries[4].texture.nextInChain = nullptr;
        entries[4].texture.sampleType = WGPUTextureSampleType_Float;
        entries[4].texture.multisampled = false;
        entries[4].texture.viewDimension = WGPUTextureViewDimension_2D;
        entries[4].storageTexture.nextInChain = nullptr;
        entries[4].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[4].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[4].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        WGPUBindGroupLayoutDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.entryCount = 5;
        descriptor.entries = entries;

        return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    }

    WGPUPipelineLayout createPipelineLayout(WGPUDevice device, std::initializer_list<WGPUBindGroupLayout> bindGroupLayouts)
    {
        WGPUPipelineLayoutDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.bindGroupLayoutCount = bindGroupLayouts.size();
        descriptor.bindGroupLayouts = &(*bindGroupLayouts.begin());

        return wgpuDeviceCreatePipelineLayout(device, &descriptor);
    }

    WGPURenderPipeline createMainPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUTextureFormat surfaceFormat)
    {
        WGPUShaderModule shaderModule = createShaderModule(device, mainShader);;

        WGPUColorTargetState colorTargetState;
        colorTargetState.nextInChain = nullptr;
        colorTargetState.format = surfaceFormat;
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

        WGPURenderPipelineDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.layout = pipelineLayout;
        descriptor.nextInChain = nullptr;
        descriptor.vertex.module = shaderModule;
        descriptor.vertex.entryPoint = "vertexMain";
        descriptor.vertex.constantCount = 0;
        descriptor.vertex.constants = nullptr;
        descriptor.vertex.bufferCount = 1;
        descriptor.vertex.buffers = &vertexBufferLayout;
        descriptor.primitive.nextInChain = nullptr;
        descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
        descriptor.primitive.frontFace = WGPUFrontFace_CCW;
        descriptor.primitive.cullMode = WGPUCullMode_None;
        descriptor.depthStencil = &depthStencilState;
        descriptor.multisample.nextInChain = nullptr;
        descriptor.multisample.count = 1;
        descriptor.multisample.mask = -1;
        descriptor.multisample.alphaToCoverageEnabled = false;
        descriptor.fragment = &fragmentState;

        return wgpuDeviceCreateRenderPipeline(device, &descriptor);
    }

    WGPUBuffer createUniformBuffer(WGPUDevice device, std::uint64_t size)
    {
        WGPUBufferDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
        descriptor.size = size;
        descriptor.mappedAtCreation = false;

        return wgpuDeviceCreateBuffer(device, &descriptor);
    }

    WGPUBindGroup createCameraBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer)
    {
        WGPUBindGroupEntry bindGroupEntry;
        bindGroupEntry.nextInChain = nullptr;
        bindGroupEntry.binding = 0;
        bindGroupEntry.buffer = uniformBuffer;
        bindGroupEntry.offset = 0;
        bindGroupEntry.size = sizeof(CameraUniform);
        bindGroupEntry.sampler = nullptr;
        bindGroupEntry.textureView = nullptr;

        WGPUBindGroupDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.layout = bindGroupLayout;
        descriptor.entryCount = 1;
        descriptor.entries = &bindGroupEntry;

        return wgpuDeviceCreateBindGroup(device, &descriptor);
    }

    WGPUBindGroup createModelBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer)
    {
        WGPUBindGroupEntry bindGroupEntry;
        bindGroupEntry.nextInChain = nullptr;
        bindGroupEntry.binding = 0;
        bindGroupEntry.buffer = uniformBuffer;
        bindGroupEntry.offset = 0;
        bindGroupEntry.size = sizeof(glm::mat4);
        bindGroupEntry.sampler = nullptr;
        bindGroupEntry.textureView = nullptr;

        WGPUBindGroupDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.layout = bindGroupLayout;
        descriptor.entryCount = 1;
        descriptor.entries = &bindGroupEntry;

        return wgpuDeviceCreateBindGroup(device, &descriptor);
    }

    WGPUTextureView createTextureView(WGPUTexture texture)
    {
        WGPUTextureViewDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.format = wgpuTextureGetFormat(texture);
        descriptor.dimension = WGPUTextureViewDimension_2D;
        descriptor.baseMipLevel = 0;
        descriptor.mipLevelCount = 1;
        descriptor.baseArrayLayer = 0;
        descriptor.arrayLayerCount = 1;
        descriptor.aspect = WGPUTextureAspect_All;

        return wgpuTextureCreateView(texture, &descriptor);
    }

    WGPUCommandEncoder createCommandEncoder(WGPUDevice device)
    {
        WGPUCommandEncoderDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;

        return wgpuDeviceCreateCommandEncoder(device, &descriptor);
    }

    WGPURenderPassEncoder createMainRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget, glm::vec4 const & clearColor)
    {
        WGPURenderPassColorAttachment colorAttachment;
        colorAttachment.nextInChain = nullptr;
        colorAttachment.view = colorTarget;
        colorAttachment.resolveTarget = nullptr;
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = {clearColor.r, clearColor.g, clearColor.b, clearColor.a};

        WGPURenderPassDepthStencilAttachment depthStencilAttachment;
        depthStencilAttachment.view = depthTarget;
        depthStencilAttachment.depthLoadOp = WGPULoadOp_Clear;
        depthStencilAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthStencilAttachment.depthClearValue = 1.f;
        depthStencilAttachment.depthReadOnly = false;
        depthStencilAttachment.stencilLoadOp = WGPULoadOp_Undefined;
        depthStencilAttachment.stencilStoreOp = WGPUStoreOp_Undefined;
        depthStencilAttachment.stencilClearValue = 0;
        depthStencilAttachment.stencilReadOnly = true;

        WGPURenderPassDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.colorAttachmentCount = 1;
        descriptor.colorAttachments = &colorAttachment;
        descriptor.depthStencilAttachment = &depthStencilAttachment;
        descriptor.occlusionQuerySet = nullptr;
        descriptor.timestampWrites = nullptr;

        return wgpuCommandEncoderBeginRenderPass(commandEncoder, &descriptor);
    }

    WGPUCommandBuffer commandEncoderFinish(WGPUCommandEncoder commandEncoder)
    {
        WGPUCommandBufferDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;

        return wgpuCommandEncoderFinish(commandEncoder, &descriptor);
    }

    WGPUSampler createDefaultSampler(WGPUDevice device)
    {
        WGPUSamplerDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.addressModeU = WGPUAddressMode_MirrorRepeat;
        descriptor.addressModeV = WGPUAddressMode_MirrorRepeat;
        descriptor.addressModeW = WGPUAddressMode_MirrorRepeat;
        descriptor.magFilter = WGPUFilterMode_Linear;
        descriptor.minFilter = WGPUFilterMode_Linear;
        descriptor.mipmapFilter = WGPUMipmapFilterMode_Linear;
        descriptor.lodMinClamp = 0.f;
        descriptor.lodMaxClamp = 255.f;
        descriptor.compare = WGPUCompareFunction_Undefined;
        descriptor.maxAnisotropy = 16;

        return wgpuDeviceCreateSampler(device, &descriptor);
    }

    WGPUTexture createWhiteTexture(WGPUDevice device, WGPUQueue queue)
    {
        auto sRGBViewFormat = WGPUTextureFormat_RGBA8UnormSrgb;

        WGPUTextureDescriptor textureDescriptor;
        textureDescriptor.nextInChain = nullptr;
        textureDescriptor.label = nullptr;
        textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
        textureDescriptor.dimension = WGPUTextureDimension_2D;
        textureDescriptor.size = {1, 1, 1};
        textureDescriptor.format = WGPUTextureFormat_RGBA8Unorm;
        textureDescriptor.mipLevelCount = 1;
        textureDescriptor.sampleCount = 1;
        textureDescriptor.viewFormatCount = 1;
        textureDescriptor.viewFormats = &sRGBViewFormat;

        auto texture = wgpuDeviceCreateTexture(device, &textureDescriptor);

        std::vector<unsigned char> pixels(4, 255);

        WGPUImageCopyTexture imageCopyTexture;
        imageCopyTexture.nextInChain = nullptr;
        imageCopyTexture.texture = texture;
        imageCopyTexture.mipLevel = 0;
        imageCopyTexture.origin = {0, 0, 0};
        imageCopyTexture.aspect = WGPUTextureAspect_All;

        WGPUTextureDataLayout textureDataLayout;
        textureDataLayout.nextInChain = nullptr;
        textureDataLayout.offset = 0;
        textureDataLayout.bytesPerRow = 4;
        textureDataLayout.rowsPerImage = 1;

        WGPUExtent3D writeSize;
        writeSize.width = 1;
        writeSize.height = 1;
        writeSize.depthOrArrayLayers = 1;

        wgpuQueueWriteTexture(queue, &imageCopyTexture, pixels.data(), pixels.size(), &textureDataLayout, &writeSize);

        return texture;
    }

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

            std::vector<std::weak_ptr<RenderObject>> users;
        };

        std::vector<std::unique_ptr<TextureInfo>> textures;

        WGPUTextureView createTextureView(std::optional<std::uint32_t> textureId, bool sRGB)
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

        ~RenderObjectCommon()
        {
            for (auto const & textureInfo : textures)
                if (auto texture = textureInfo->texture.load())
                    wgpuTextureRelease(texture);

            wgpuBufferRelease(indexBuffer);
            wgpuBufferRelease(vertexBuffer);
        }
    };

}

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

    glm::mat4 modelMatrix;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;

    struct Material
    {
        MaterialUniform uniforms;

        std::optional<std::uint32_t> baseColorTextureId;
        std::optional<std::uint32_t> metallicRoughnessTextureId;
        std::optional<std::uint32_t> normalTextureId;
    };

    Material material;

    WGPUBindGroup materialBindGroup;
    std::atomic<bool> texturesUpdated{false};

    void createMaterialBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, WGPUBuffer materialUniformBuffer, WGPUSampler sampler)
    {
        WGPUBindGroupEntry materialBindGroupEntries[5];

        materialBindGroupEntries[0].nextInChain = nullptr;
        materialBindGroupEntries[0].binding = 0;
        materialBindGroupEntries[0].buffer = materialUniformBuffer;
        materialBindGroupEntries[0].offset = 0;
        materialBindGroupEntries[0].size = sizeof(MaterialUniform);
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
        materialBindGroupEntries[2].textureView = common->createTextureView(material.baseColorTextureId, true);

        materialBindGroupEntries[3].nextInChain = nullptr;
        materialBindGroupEntries[3].binding = 3;
        materialBindGroupEntries[3].buffer = nullptr;
        materialBindGroupEntries[3].offset = 0;
        materialBindGroupEntries[3].size = 0;
        materialBindGroupEntries[3].sampler = nullptr;
        materialBindGroupEntries[3].textureView = common->createTextureView(material.normalTextureId, false);

        materialBindGroupEntries[4].nextInChain = nullptr;
        materialBindGroupEntries[4].binding = 4;
        materialBindGroupEntries[4].buffer = nullptr;
        materialBindGroupEntries[4].offset = 0;
        materialBindGroupEntries[4].size = 0;
        materialBindGroupEntries[4].sampler = nullptr;
        materialBindGroupEntries[4].textureView = common->createTextureView(material.metallicRoughnessTextureId, false);

        WGPUBindGroupDescriptor materialBindGroupDescriptor;
        materialBindGroupDescriptor.nextInChain = nullptr;
        materialBindGroupDescriptor.label = nullptr;
        materialBindGroupDescriptor.layout = layout;
        materialBindGroupDescriptor.entryCount = 5;
        materialBindGroupDescriptor.entries = materialBindGroupEntries;

        materialBindGroup = wgpuDeviceCreateBindGroup(device, &materialBindGroupDescriptor);
    }

    ~RenderObject()
    {
        if (materialBindGroup)
            wgpuBindGroupRelease(materialBindGroup);
    }
};

namespace
{

    struct AsyncLoader
    {
        AsyncLoader(WGPUDevice device, WGPUQueue queue)
            : device_(device)
            , queue_(queue)
        {}

        struct TextureLoadTask
        {
            std::shared_ptr<RenderObjectCommon> common;
            std::uint32_t textureId;
        };

        void push(TextureLoadTask const & task)
        {
            {
                std::lock_guard lock{tasksMutex_};
                tasks_.push_back(task);
            }
            tasksCV_.notify_one();
        }

        void stop()
        {
            {
                std::lock_guard lock{tasksMutex_};
                tasks_.clear();
                tasks_.push_back(TextureLoadTask{nullptr, 0});
            }
            tasksCV_.notify_one();
        }

        void run()
        {
            while (true)
            {
                TextureLoadTask task{nullptr, 0};

                {
                    std::unique_lock lock{tasksMutex_};
                    tasksCV_.wait(lock, [&]{ return !tasks_.empty(); });
                    task = tasks_.front();
                    tasks_.pop_front();
                }

                if (!task.common)
                    break;

                auto & textureInfo = *task.common->textures[task.textureId];

                auto imageInfo = glTF::loadImage(textureInfo.assetPath, textureInfo.uri);

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

                textureDescriptor.mipLevelCount = std::floor(std::log2(std::max(imageInfo.width, imageInfo.height))) + 1;
                textureDescriptor.sampleCount = 1;
                textureDescriptor.viewFormatCount = sRGBViewFormat ? 1 : 0;
                textureDescriptor.viewFormats = sRGBViewFormat ? &(*sRGBViewFormat) : nullptr;

                auto texture = wgpuDeviceCreateTexture(device_, &textureDescriptor);

                std::vector<unsigned char> levelPixels;
                int levelWidth = imageInfo.width;
                int levelHeight = imageInfo.height;

                for (int i = 0; i < textureDescriptor.mipLevelCount; ++i)
                {
                    if (levelPixels.empty())
                    {
                        levelPixels.assign(imageInfo.data.get(), imageInfo.data.get() + imageInfo.width * imageInfo.height * imageInfo.channels);
                    }
                    else
                    {
                        int newLevelWidth = levelWidth / 2;
                        int newLevelHeight = levelHeight / 2;
                        std::vector<unsigned char> newLevelPixels(newLevelWidth * newLevelHeight * imageInfo.channels, 0);

                        for (int y = 0; y < newLevelHeight; ++y)
                        {
                            for (int x = 0; x < newLevelWidth; ++x)
                            {
                                for (int c = 0; c < imageInfo.channels; ++c)
                                {
                                    int sum = 0;
                                    sum += levelPixels[((2 * y + 0) * levelWidth + (2 * x + 0)) * imageInfo.channels + c];
                                    sum += levelPixels[((2 * y + 0) * levelWidth + (2 * x + 1)) * imageInfo.channels + c];
                                    sum += levelPixels[((2 * y + 1) * levelWidth + (2 * x + 0)) * imageInfo.channels + c];
                                    sum += levelPixels[((2 * y + 1) * levelWidth + (2 * x + 1)) * imageInfo.channels + c];
                                    newLevelPixels[(y * newLevelWidth + x) * imageInfo.channels + c] = sum >> 2;
                                }
                            }
                        }

                        levelHeight = newLevelHeight;
                        levelWidth = newLevelWidth;
                        levelPixels = std::move(newLevelPixels);
                    }

                    WGPUImageCopyTexture imageCopyTexture;
                    imageCopyTexture.nextInChain = nullptr;
                    imageCopyTexture.texture = texture;
                    imageCopyTexture.mipLevel = i;
                    imageCopyTexture.origin = {0, 0, 0};
                    imageCopyTexture.aspect = WGPUTextureAspect_All;

                    WGPUTextureDataLayout textureDataLayout;
                    textureDataLayout.nextInChain = nullptr;
                    textureDataLayout.offset = 0;
                    textureDataLayout.bytesPerRow = levelWidth * imageInfo.channels;
                    textureDataLayout.rowsPerImage = levelHeight;

                    WGPUExtent3D writeSize;
                    writeSize.width = levelWidth;
                    writeSize.height = levelHeight;
                    writeSize.depthOrArrayLayers = 1;

                    wgpuQueueWriteTexture(queue_, &imageCopyTexture, levelPixels.data(), levelPixels.size(), &textureDataLayout, &writeSize);
                }

                textureInfo.texture.store(texture);

                for (auto const & user : textureInfo.users)
                    if (auto object = user.lock())
                        object->texturesUpdated.store(true);
            }
        }

    private:
        WGPUDevice device_;
        WGPUQueue queue_;

        std::mutex tasksMutex_;
        std::deque<TextureLoadTask> tasks_;
        std::condition_variable tasksCV_;
    };

}

struct Engine::Impl
{
    Impl(WGPUDevice device, WGPUQueue queue);
    ~Impl();

    void render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera);
    std::vector<RenderObjectPtr> loadGLTF(std::filesystem::path const & assetPath);

private:
    WGPUDevice device_;
    WGPUQueue queue_;

    AsyncLoader loader_;
    std::thread loaderThread_;

    WGPUBindGroupLayout cameraBindGroupLayout_;
    WGPUBindGroupLayout modelBindGroupLayout_;
    WGPUBindGroupLayout materialBindGroupLayout_;

    WGPUPipelineLayout mainPipelineLayout_;
    WGPURenderPipeline mainPipeline_;

    WGPUBuffer cameraUniformBuffer_;
    WGPUBuffer modelUniformBuffer_;
    WGPUBuffer materialUniformBuffer_;

    WGPUBindGroup cameraBindGroup_;
    WGPUBindGroup modelBindGroup_;

    WGPUTexture depthTexture_;
    WGPUTextureView depthTextureView_;

    WGPUSampler defaultSampler_;
    WGPUTexture whiteTexture_;

    glm::uvec2 cachedRenderTargetSize_{0, 0};

    void resizeDepthBuffer(glm::uvec2 const & renderTargetSize);
    void updateCameraUniformBuffer(Camera const & camera);
};

Engine::Impl::Impl(WGPUDevice device, WGPUQueue queue)
    : device_(device)
    , queue_(queue)
    , loader_(device_, queue_)
    , loaderThread_([this]{ loader_.run(); })
    , cameraBindGroupLayout_(createCameraBindGroupLayout(device_))
    , modelBindGroupLayout_(createModelBindGroupLayout(device_))
    , materialBindGroupLayout_(createMaterialBindGroupLayout(device_))
    , mainPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, modelBindGroupLayout_, materialBindGroupLayout_}))
    , mainPipeline_(nullptr)
    , cameraUniformBuffer_(createUniformBuffer(device_, sizeof(CameraUniform)))
    , modelUniformBuffer_(createUniformBuffer(device_, sizeof(glm::mat4)))
    , materialUniformBuffer_(createUniformBuffer(device_, sizeof(MaterialUniform)))
    , cameraBindGroup_(createCameraBindGroup(device_, cameraBindGroupLayout_, cameraUniformBuffer_))
    , modelBindGroup_(createModelBindGroup(device_, modelBindGroupLayout_, modelUniformBuffer_))
    , depthTexture_(nullptr)
    , depthTextureView_(nullptr)
    , defaultSampler_(createDefaultSampler(device_))
    , whiteTexture_(createWhiteTexture(device_, queue_))
{}

Engine::Impl::~Impl()
{
    loader_.push({nullptr, 0});
    loaderThread_.join();

    wgpuTextureRelease(whiteTexture_);
    wgpuSamplerRelease(defaultSampler_);
    wgpuTextureViewRelease(depthTextureView_);
    wgpuTextureRelease(depthTexture_);
    wgpuBindGroupRelease(modelBindGroup_);
    wgpuBindGroupRelease(cameraBindGroup_);
    wgpuBufferRelease(materialUniformBuffer_);
    wgpuBufferRelease(modelUniformBuffer_);
    wgpuBufferRelease(cameraUniformBuffer_);
    wgpuRenderPipelineRelease(mainPipeline_);
    wgpuPipelineLayoutRelease(mainPipelineLayout_);
    wgpuBindGroupLayoutRelease(materialBindGroupLayout_);
    wgpuBindGroupLayoutRelease(modelBindGroupLayout_);
    wgpuBindGroupLayoutRelease(cameraBindGroupLayout_);
}

void Engine::Impl::render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera)
{
    if (!mainPipeline_)
        mainPipeline_ = createMainPipeline(device_, mainPipelineLayout_, wgpuTextureGetFormat(target));

    resizeDepthBuffer({wgpuTextureGetWidth(target), wgpuTextureGetHeight(target)});
    updateCameraUniformBuffer(camera);

    WGPUTextureView targetView = createTextureView(target);
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPassEncoder = createMainRenderPass(commandEncoder, targetView, depthTextureView_, {0.8f, 0.9f, 1.f, 1.f});
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, mainPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, cameraBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, modelBindGroup_, 0, nullptr);

    for (auto const & object : objects)
    {
        wgpuQueueWriteBuffer(queue_, modelUniformBuffer_, 0, &object->modelMatrix, 64);
        wgpuQueueWriteBuffer(queue_, materialUniformBuffer_, 0, &object->material.uniforms, sizeof(MaterialUniform));

        // TODO: need compare-exchange here
        if (object->texturesUpdated.load())
        {
            object->texturesUpdated.store(false);
            object->createMaterialBindGroup(device_, materialBindGroupLayout_, materialUniformBuffer_, defaultSampler_);
        }

        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, object->materialBindGroup, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 0, object->common->vertexBuffer, object->vertexByteOffset, object->vertexByteLength);
        wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder, object->common->indexBuffer, object->indexFormat, object->indexByteOffset, object->indexByteLength);
        wgpuRenderPassEncoderDrawIndexed(renderPassEncoder, object->indexCount, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(renderPassEncoder);

    auto commandBuffer = commandEncoderFinish(commandEncoder);

    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuTextureViewRelease(targetView);
}

std::vector<RenderObjectPtr> Engine::Impl::loadGLTF(std::filesystem::path const & assetPath)
{
    glTF::Asset asset = glTF::load(assetPath);

    auto common = std::make_shared<RenderObjectCommon>();

    common->whiteTexture = whiteTexture_;

    for (auto const & textureIn : asset.textures)
    {
        auto & texture = *common->textures.emplace_back(std::make_unique<RenderObjectCommon::TextureInfo>());

        if (!textureIn.source) continue;

        texture.assetPath = assetPath;
        texture.uri = asset.images[*textureIn.source].uri;
    }

    std::vector<RenderObjectPtr> result;

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

                auto & renderObject = result.emplace_back(std::make_shared<RenderObject>());

                renderObject->common = common;

                renderObject->vertexByteOffset = vertices.size() * sizeof(vertices[0]);
                renderObject->vertexByteLength = positionAccessor.count * sizeof(vertices[0]);
                renderObject->vertexCount = positionAccessor.count;
                renderObject->indexByteOffset = indices.size() * sizeof(indices[0]);
                renderObject->indexByteLength = indexAccessor.count * sizeof(indices[0]);
                renderObject->indexCount = indexAccessor.count;
                renderObject->indexFormat = WGPUIndexFormat_Uint16;
                renderObject->modelMatrix = nodeModelMatrix;
                renderObject->bboxMin = glm::vec3( std::numeric_limits<float>::infinity());
                renderObject->bboxMax = glm::vec3(-std::numeric_limits<float>::infinity());

                renderObject->material.uniforms.baseColorFactor = materialIn.baseColorFactor;
                renderObject->material.uniforms.metallicFactor = materialIn.metallicFactor;
                renderObject->material.uniforms.roughnessFactor = materialIn.roughnessFactor;
                renderObject->material.uniforms.emissiveFactor = materialIn.emissiveFactor;

                renderObject->material.baseColorTextureId = materialIn.baseColorTexture;
                renderObject->material.metallicRoughnessTextureId = materialIn.metallicRoughnessTexture;
                renderObject->material.normalTextureId = materialIn.normalTexture;

                if (materialIn.baseColorTexture)
                    common->textures[*materialIn.baseColorTexture]->users.push_back(renderObject);
                if (materialIn.metallicRoughnessTexture)
                    common->textures[*materialIn.metallicRoughnessTexture]->users.push_back(renderObject);
                if (materialIn.normalTexture)
                    common->textures[*materialIn.normalTexture]->users.push_back(renderObject);

                auto const & positionBufferView = asset.bufferViews[positionAccessor.bufferView];
                auto const &   normalBufferView = asset.bufferViews[  normalAccessor.bufferView];
                auto const &  tangentBufferView = asset.bufferViews[ tangentAccessor.bufferView];
                auto const & texcoordBufferView = asset.bufferViews[texcoordAccessor.bufferView];
                auto const &    indexBufferView = asset.bufferViews[   indexAccessor.bufferView];

                auto positionIterator = glTF::AccessorIterator<glm::vec3>(assetBufferData.data() + positionBufferView.byteOffset + positionAccessor.byteOffset, positionBufferView.byteStride);
                auto normalIterator = glTF::AccessorIterator<glm::vec3>(assetBufferData.data() + normalBufferView.byteOffset + normalAccessor.byteOffset, normalBufferView.byteStride);
                auto tangentIterator = glTF::AccessorIterator<glm::vec4>(assetBufferData.data() + tangentBufferView.byteOffset + tangentAccessor.byteOffset, tangentBufferView.byteStride);
                auto texcoordIterator = glTF::AccessorIterator<glm::vec2>(assetBufferData.data() + texcoordBufferView.byteOffset + texcoordAccessor.byteOffset, texcoordBufferView.byteStride);

                for (int i = 0; i < positionAccessor.count; ++i)
                {
                    vertices.push_back({
                        *positionIterator++,
                        *  normalIterator++,
                        * tangentIterator++,
                        *texcoordIterator++,
                    });

                    auto transformedVertex = glm::vec3((renderObject->modelMatrix * glm::vec4(vertices.back().position, 1.f)));

                    renderObject->bboxMin = glm::min(renderObject->bboxMin, transformedVertex);
                    renderObject->bboxMax = glm::max(renderObject->bboxMax, transformedVertex);
                }

                auto indexIterator = glTF::AccessorIterator<std::uint16_t>(assetBufferData.data() + indexBufferView.byteOffset + indexAccessor.byteOffset, indexBufferView.byteStride);
                for (int i = 0; i < indexAccessor.count; ++i)
                    indices.emplace_back(*indexIterator++);

                renderObject->texturesUpdated = true;
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

        common->vertexBuffer = wgpuDeviceCreateBuffer(device_, &vertexBufferDescriptor);

        wgpuQueueWriteBuffer(queue_, common->vertexBuffer, 0, vertices.data(), vertices.size() * sizeof(vertices[0]));

        WGPUBufferDescriptor indexBufferDescriptor;
        indexBufferDescriptor.nextInChain = nullptr;
        indexBufferDescriptor.label = nullptr;
        indexBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index;
        indexBufferDescriptor.size = indices.size() * sizeof(indices[0]);
        indexBufferDescriptor.mappedAtCreation = false;

        common->indexBuffer = wgpuDeviceCreateBuffer(device_, &indexBufferDescriptor);

        wgpuQueueWriteBuffer(queue_, common->indexBuffer, 0, indices.data(), indices.size() * sizeof(indices[0]));
    }

    for (std::uint32_t i = 0; i < common->textures.size(); ++i)
        loader_.push({common, i});

    return result;
}

void Engine::Impl::resizeDepthBuffer(glm::uvec2 const & renderTargetSize)
{
    if (!depthTexture_ || cachedRenderTargetSize_ != renderTargetSize)
    {
        if (depthTexture_)
        {
            wgpuTextureViewRelease(depthTextureView_);
            wgpuTextureRelease(depthTexture_);
        }

        WGPUTextureDescriptor depthTextureDescriptor;
        depthTextureDescriptor.nextInChain = nullptr;
        depthTextureDescriptor.label = nullptr;
        depthTextureDescriptor.usage = WGPUTextureUsage_RenderAttachment;
        depthTextureDescriptor.dimension = WGPUTextureDimension_2D;
        depthTextureDescriptor.size = {renderTargetSize.x, renderTargetSize.y, 1};
        depthTextureDescriptor.format = WGPUTextureFormat_Depth24Plus;
        depthTextureDescriptor.mipLevelCount = 1;
        depthTextureDescriptor.sampleCount = 1;
        depthTextureDescriptor.viewFormatCount = 1;
        depthTextureDescriptor.viewFormats = &depthTextureDescriptor.format;

        depthTexture_ = wgpuDeviceCreateTexture(device_, &depthTextureDescriptor);
        depthTextureView_ = createTextureView(depthTexture_);

        cachedRenderTargetSize_ = renderTargetSize;
    }
}

void Engine::Impl::updateCameraUniformBuffer(Camera const & camera)
{
    CameraUniform cameraUniform;
    cameraUniform.viewProjection = camera.viewProjectionMatrix();
    cameraUniform.viewProjectionInverse = glm::inverse(cameraUniform.viewProjection);
    cameraUniform.position = camera.position();

    wgpuQueueWriteBuffer(queue_, cameraUniformBuffer_, 0, &cameraUniform, sizeof(CameraUniform));
}

Engine::Engine(WGPUDevice device, WGPUQueue queue)
    : pimpl_(std::make_unique<Impl>(device, queue))
{}

Engine::~Engine() = default;

std::vector<RenderObjectPtr> Engine::loadGLTF(std::filesystem::path const & assetPath)
{
    return pimpl_->loadGLTF(assetPath);
}

void Engine::render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera)
{
    pimpl_->render(target, objects, camera);
}

std::pair<glm::vec3, glm::vec3> Engine::bbox(std::vector<RenderObjectPtr> const & objects) const
{
    glm::vec3 min(  std::numeric_limits<float>::infinity());
    glm::vec3 max(- std::numeric_limits<float>::infinity());

    for (auto const & object : objects)
    {
        min = glm::min(min, object->bboxMin);
        max = glm::max(max, object->bboxMax);
    }

    return {min, max};
}
