#include <webgpu-demo/engine.hpp>
#include <webgpu-demo/gltf_loader.hpp>
#include <webgpu-demo/gltf_iterator.hpp>
#include <webgpu-demo/synchronized_queue.hpp>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <atomic>
#include <thread>
#include <functional>

namespace
{

    glm::mat4 glToVkProjection(glm::mat4 matrix)
    {
        // Map from [-1, 1] z-range in OpenGL to [0, 1] z-range in Vulkan, WebGPU, etc
        // Equivalent to doing v.z = (v.z + v.w) / 2 after applying the matrix

        for (int i = 0; i < 4; ++i)
            matrix[i][2] = (matrix[i][2] + matrix[i][3]) / 2.f;

        return matrix;
    }

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

    struct ObjectUniform
    {
        glm::mat4 modelMatrix;
        glm::vec4 baseColorFactor;
        float metallicFactor;
        float roughnessFactor;
        float padding1[2];
        glm::vec3 emissiveFactor;
        float padding2[1];
    };

    struct LightsUniform
    {
        glm::mat4 shadowProjection;
        glm::vec3 ambientLight;
        float padding1[1];
        glm::vec3 sunDirection;
        float padding2[1];
        glm::vec3 sunIntensity;
        float padding3[1];
    };

    static const char mainShader[] =
R"(

struct Camera {
    viewProjection : mat4x4f,
    viewProjectionInverse : mat4x4f,
    position : vec3f,
}

struct Object {
    model : mat4x4f,
    baseColorFactor : vec4f,
    metallicFactor : f32,
    roughnessFactor : f32,
    emissiveFactor : vec3f,
}

struct Lights {
    shadowProjection : mat4x4f,
    ambientLight : vec3f,
    sunDirection : vec3f,
    sunIntensity : vec3f,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform> object: Object;

@group(2) @binding(0) var textureSampler: sampler;
@group(2) @binding(1) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(2) var normalTexture: texture_2d<f32>;
@group(2) @binding(3) var metallicRoughnessTexture: texture_2d<f32>;

@group(3) @binding(0) var<uniform> lights : Lights;
@group(3) @binding(1) var shadowSampler: sampler_comparison;
@group(3) @binding(2) var shadowMapTexture: texture_depth_2d;

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
    let worldPosition = (object.model * vec4f(in.position, 1.0)).xyz;
    let position : vec4f = camera.viewProjection * vec4f(worldPosition, 1.0);
    let normal : vec3f = normalize(asMat3x3(object.model) * in.normal);
    let tangent : vec4f = vec4f(normalize(asMat3x3(object.model) * in.tangent.xyz), in.tangent.w);
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
    let baseColorSample = textureSample(baseColorTexture, textureSampler, in.texcoord) * object.baseColorFactor;

    let baseColor = baseColorSample.rgb;

    var tbn = mat3x3f();
    tbn[2] = normalize(in.normal);
    tbn[0] = normalize(in.tangent.xyz - tbn[2] * dot(in.tangent.xyz, tbn[2]));
    tbn[1] = cross(tbn[2], tbn[0]) * in.tangent.w;

    let normal = tbn * normalize(2.0 * textureSample(normalTexture, textureSampler, in.texcoord).rgb - vec3(1.0));

    let materialSample = textureSample(metallicRoughnessTexture, textureSampler, in.texcoord);

    let metallic = materialSample.b * object.metallicFactor;
    let roughness = materialSample.g * object.roughnessFactor;

    let viewDirection = normalize(camera.position - in.worldPosition);
    let halfway = normalize(lights.sunDirection + viewDirection);

    let lightness = max(0.0, dot(normal, lights.sunDirection));
    let specular = lightness * metallic * pow(max(0.0, dot(viewDirection, halfway)), 1.0 / max(0.0001, roughness * roughness));

    let shadowPositionClip = lights.shadowProjection * vec4(in.worldPosition, 1.0);

    let shadowPositionNdc = shadowPositionClip.xyz / shadowPositionClip.w;

    let shadowBias = 0.001;

    let shadowFactor = textureSampleCompare(shadowMapTexture, shadowSampler, shadowPositionNdc.xy * vec2f(0.5, -0.5) + vec2f(0.5), shadowPositionNdc.z - shadowBias);

    let outColor = (lights.ambientLight + (lightness + specular) * shadowFactor * lights.sunIntensity) * baseColor;

    return vec4f(tonemap(outColor), baseColorSample.a);
}

struct ShadowVertexInput {
    @location(0) position : vec3f,
    @location(1) texcoord : vec2f,
}

struct ShadowVertexOutput {
    @builtin(position) position : vec4f,
    @location(0) texcoord : vec2f,
}

@vertex
fn shadowVertexMain(in : ShadowVertexInput) -> ShadowVertexOutput {
    let position : vec4f = camera.viewProjection * object.model * vec4f(in.position, 1.0);
    return ShadowVertexOutput(position, in.texcoord);
}

@fragment
fn shadowFragmentMain(in : ShadowVertexOutput) {
    let baseColorSample = textureSample(baseColorTexture, textureSampler, in.texcoord);

    if (baseColorSample.a < 0.5) {
        discard;
    }
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

    WGPUBindGroupLayout createObjectBindGroupLayout(WGPUDevice device)
    {
        WGPUBindGroupLayoutEntry entries[1];

        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        entries[0].buffer.nextInChain = nullptr;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.hasDynamicOffset = true;
        entries[0].buffer.minBindingSize = sizeof(ObjectUniform);
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

    WGPUBindGroupLayout createTexturesBindGroupLayout(WGPUDevice device)
    {
        WGPUBindGroupLayoutEntry entries[4];

        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Fragment;
        entries[0].buffer.nextInChain = nullptr;
        entries[0].buffer.type = WGPUBufferBindingType_Undefined;
        entries[0].buffer.hasDynamicOffset = false;
        entries[0].buffer.minBindingSize = 0;
        entries[0].sampler.nextInChain = nullptr;
        entries[0].sampler.type = WGPUSamplerBindingType_Filtering;
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
        entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
        entries[1].texture.nextInChain = nullptr;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.multisampled = false;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
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

        WGPUBindGroupLayoutDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.entryCount = 4;
        descriptor.entries = entries;

        return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    }

    WGPUBindGroupLayout createLightsBindGroupLayout(WGPUDevice device)
    {
        WGPUBindGroupLayoutEntry entries[3];

        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Fragment;
        entries[0].buffer.nextInChain = nullptr;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.hasDynamicOffset = false;
        entries[0].buffer.minBindingSize = sizeof(LightsUniform);
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
        entries[1].sampler.type = WGPUSamplerBindingType_Comparison;
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
        entries[2].texture.sampleType = WGPUTextureSampleType_Depth;
        entries[2].texture.multisampled = false;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
        entries[2].storageTexture.nextInChain = nullptr;
        entries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
        entries[2].storageTexture.format = WGPUTextureFormat_Undefined;
        entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

        WGPUBindGroupLayoutDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.entryCount = 3;
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

    WGPURenderPipeline createMainPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUTextureFormat surfaceFormat, WGPUShaderModule shaderModule)
    {
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
        descriptor.label = "main";
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
        descriptor.multisample.count = 4;
        descriptor.multisample.mask = -1;
        descriptor.multisample.alphaToCoverageEnabled = true;
        descriptor.fragment = &fragmentState;

        return wgpuDeviceCreateRenderPipeline(device, &descriptor);
    }

    WGPURenderPipeline createShadowPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
    {
        WGPUFragmentState fragmentState;
        fragmentState.nextInChain = nullptr;
        fragmentState.module = shaderModule;
        fragmentState.entryPoint = "shadowFragmentMain";
        fragmentState.constantCount = 0;
        fragmentState.constants = nullptr;
        fragmentState.targetCount = 0;
        fragmentState.targets = nullptr;

        WGPUVertexAttribute attributes[2];
        attributes[0].format = WGPUVertexFormat_Float32x3;
        attributes[0].offset = 0;
        attributes[0].shaderLocation = 0;
        attributes[1].format = WGPUVertexFormat_Float32x2;
        attributes[1].offset = 40;
        attributes[1].shaderLocation = 1;

        WGPUVertexBufferLayout vertexBufferLayout;
        vertexBufferLayout.arrayStride = sizeof(Vertex);
        vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
        vertexBufferLayout.attributeCount = 2;
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
        descriptor.label = "shadow";
        descriptor.layout = pipelineLayout;
        descriptor.nextInChain = nullptr;
        descriptor.vertex.module = shaderModule;
        descriptor.vertex.entryPoint = "shadowVertexMain";
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

    WGPUBindGroup createObjectBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer)
    {
        WGPUBindGroupEntry entries[1];

        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].buffer = uniformBuffer;
        entries[0].offset = 0;
        entries[0].size = sizeof(ObjectUniform);
        entries[0].sampler = nullptr;
        entries[0].textureView = nullptr;

        WGPUBindGroupDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.layout = bindGroupLayout;
        descriptor.entryCount = 1;
        descriptor.entries = entries;

        return wgpuDeviceCreateBindGroup(device, &descriptor);
    }

    WGPUBindGroup createLightsBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer, WGPUSampler shadowSampler, WGPUTextureView shadowMapView)
    {
        WGPUBindGroupEntry entries[3];

        entries[0].nextInChain = nullptr;
        entries[0].binding = 0;
        entries[0].buffer = uniformBuffer;
        entries[0].offset = 0;
        entries[0].size = sizeof(LightsUniform);
        entries[0].sampler = nullptr;
        entries[0].textureView = nullptr;

        entries[1].nextInChain = nullptr;
        entries[1].binding = 1;
        entries[1].buffer = nullptr;
        entries[1].offset = 0;
        entries[1].size = 0;
        entries[1].sampler = shadowSampler;
        entries[1].textureView = nullptr;

        entries[2].nextInChain = nullptr;
        entries[2].binding = 2;
        entries[2].buffer = nullptr;
        entries[2].offset = 0;
        entries[2].size = 0;
        entries[2].sampler = nullptr;
        entries[2].textureView = shadowMapView;

        WGPUBindGroupDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.layout = bindGroupLayout;
        descriptor.entryCount = 3;
        descriptor.entries = entries;

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

    WGPURenderPassEncoder createMainRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget, WGPUTextureView resolveTarget, glm::vec4 const & clearColor)
    {
        WGPURenderPassColorAttachment colorAttachment;
        colorAttachment.nextInChain = nullptr;
        colorAttachment.view = colorTarget;
        colorAttachment.resolveTarget = resolveTarget;
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

    WGPURenderPassEncoder createShadowRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView depthTarget)
    {
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
        descriptor.colorAttachmentCount = 0;
        descriptor.colorAttachments = nullptr;
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

    WGPUSampler createShadowSampler(WGPUDevice device)
    {
        WGPUSamplerDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.addressModeU = WGPUAddressMode_MirrorRepeat;
        descriptor.addressModeV = WGPUAddressMode_MirrorRepeat;
        descriptor.addressModeW = WGPUAddressMode_MirrorRepeat;
        descriptor.magFilter = WGPUFilterMode_Linear;
        descriptor.minFilter = WGPUFilterMode_Linear;
        descriptor.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        descriptor.lodMinClamp = 0.f;
        descriptor.lodMaxClamp = 0.f;
        descriptor.compare = WGPUCompareFunction_LessEqual;
        descriptor.maxAnisotropy = 1;

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

    WGPUTexture createShadowMapTexture(WGPUDevice device, std::uint32_t size)
    {
        WGPUTextureDescriptor textureDescriptor;
        textureDescriptor.nextInChain = nullptr;
        textureDescriptor.label = nullptr;
        textureDescriptor.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment;
        textureDescriptor.dimension = WGPUTextureDimension_2D;
        textureDescriptor.size = {size, size, 1};
        textureDescriptor.format = WGPUTextureFormat_Depth24Plus;
        textureDescriptor.mipLevelCount = 1;
        textureDescriptor.sampleCount = 1;
        textureDescriptor.viewFormatCount = 0;
        textureDescriptor.viewFormats = nullptr;

        return wgpuDeviceCreateTexture(device, &textureDescriptor);
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

    void createTexturesBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, WGPUSampler sampler)
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
    }

    ~RenderObject()
    {
        if (texturesBindGroup)
            wgpuBindGroupRelease(texturesBindGroup);
    }
};

struct Engine::Impl
{
    Impl(WGPUDevice device, WGPUQueue queue);
    ~Impl();

    void render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, LightSettings const & lightSettings);
    std::vector<RenderObjectPtr> loadGLTF(std::filesystem::path const & assetPath);

private:
    WGPUDevice device_;
    WGPUQueue queue_;

    using Task = std::function<void()>;

    SynchronizedQueue<Task> loaderQueue_;
    SynchronizedQueue<Task> renderQueue_;
    std::thread loaderThread_;

    WGPUBindGroupLayout cameraBindGroupLayout_;
    WGPUBindGroupLayout objectBindGroupLayout_;
    WGPUBindGroupLayout texturesBindGroupLayout_;
    WGPUBindGroupLayout lightsBindGroupLayout_;

    WGPUShaderModule shaderModule_;

    WGPUTexture shadowMap_;
    WGPUTextureView shadowMapView_;

    WGPUSampler defaultSampler_;
    WGPUSampler shadowSampler_;

    WGPUPipelineLayout mainPipelineLayout_;
    WGPURenderPipeline mainPipeline_;
    WGPUPipelineLayout shadowPipelineLayout_;
    WGPURenderPipeline shadowPipeline_;

    WGPUBuffer cameraUniformBuffer_;
    WGPUBuffer objectUniformBuffer_;
    WGPUBuffer lightsUniformBuffer_;

    WGPUBindGroup cameraBindGroup_;
    WGPUBindGroup objectBindGroup_;
    WGPUBindGroup lightsBindGroup_;

    WGPUTexture frameTexture_;
    WGPUTextureView frameTextureView_;
    WGPUTexture depthTexture_;
    WGPUTextureView depthTextureView_;

    WGPUTexture whiteTexture_;

    glm::uvec2 cachedRenderTargetSize_{0, 0};

    void updateFrameBuffer(glm::uvec2 const & renderTargetSize, WGPUTextureFormat surfaceFormat);
    void updateCameraUniformBuffer(Camera const & camera);
    glm::mat4 computeShadowProjection(glm::vec3 const & lightDirection, Box const & sceneBbox);
    void updateCameraUniformBufferShadow(glm::mat4 const & shadowProjection);
    void updateShadowUniformBuffer(glm::mat4 const & shadowProjection, LightSettings const & lightSettings);
    void loadTexture(RenderObjectCommon::TextureInfo & textureInfo);
    void loaderThreadMain();
};

Engine::Impl::Impl(WGPUDevice device, WGPUQueue queue)
    : device_(device)
    , queue_(queue)
    , loaderThread_([this]{ loaderThreadMain(); })
    , cameraBindGroupLayout_(createCameraBindGroupLayout(device_))
    , objectBindGroupLayout_(createObjectBindGroupLayout(device_))
    , texturesBindGroupLayout_(createTexturesBindGroupLayout(device_))
    , lightsBindGroupLayout_(createLightsBindGroupLayout(device_))
    , shaderModule_(createShaderModule(device_, mainShader))
    , shadowMap_(createShadowMapTexture(device_, 4096))
    , shadowMapView_(createTextureView(shadowMap_))
    , defaultSampler_(createDefaultSampler(device_))
    , shadowSampler_(createShadowSampler(device_))
    , mainPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, objectBindGroupLayout_, texturesBindGroupLayout_, lightsBindGroupLayout_}))
    , mainPipeline_(nullptr)
    , shadowPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, objectBindGroupLayout_, texturesBindGroupLayout_}))
    , shadowPipeline_(createShadowPipeline(device_, shadowPipelineLayout_, shaderModule_))
    , cameraUniformBuffer_(createUniformBuffer(device_, sizeof(CameraUniform)))
    , objectUniformBuffer_(nullptr)
    , lightsUniformBuffer_(createUniformBuffer(device_, sizeof(LightsUniform)))
    , cameraBindGroup_(createCameraBindGroup(device_, cameraBindGroupLayout_, cameraUniformBuffer_))
    , objectBindGroup_(nullptr)
    , lightsBindGroup_(createLightsBindGroup(device_, lightsBindGroupLayout_, lightsUniformBuffer_, shadowSampler_, shadowMapView_))
    , frameTexture_(nullptr)
    , frameTextureView_(nullptr)
    , depthTexture_(nullptr)
    , depthTextureView_(nullptr)
    , whiteTexture_(createWhiteTexture(device_, queue_))
{}

Engine::Impl::~Impl()
{
    loaderQueue_.grab();
    loaderQueue_.push(nullptr);
    loaderThread_.join();

    wgpuTextureRelease(whiteTexture_);
    wgpuTextureViewRelease(depthTextureView_);
    wgpuTextureRelease(depthTexture_);
    wgpuTextureViewRelease(frameTextureView_);
    wgpuTextureRelease(frameTexture_);
    wgpuBindGroupRelease(lightsBindGroup_);
    wgpuBindGroupRelease(objectBindGroup_);
    wgpuBindGroupRelease(cameraBindGroup_);
    wgpuBufferRelease(lightsUniformBuffer_);
    wgpuBufferRelease(objectUniformBuffer_);
    wgpuBufferRelease(cameraUniformBuffer_);
    wgpuRenderPipelineRelease(shadowPipeline_);
    wgpuPipelineLayoutRelease(shadowPipelineLayout_);
    wgpuRenderPipelineRelease(mainPipeline_);
    wgpuPipelineLayoutRelease(mainPipelineLayout_);
    wgpuSamplerRelease(shadowSampler_);
    wgpuSamplerRelease(defaultSampler_);
    wgpuTextureViewRelease(shadowMapView_);
    wgpuTextureRelease(shadowMap_);
    wgpuShaderModuleRelease(shaderModule_);
    wgpuBindGroupLayoutRelease(lightsBindGroupLayout_);
    wgpuBindGroupLayoutRelease(texturesBindGroupLayout_);
    wgpuBindGroupLayoutRelease(objectBindGroupLayout_);
    wgpuBindGroupLayoutRelease(cameraBindGroupLayout_);
}

void Engine::Impl::render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, LightSettings const & lightSettings)
{
    for (auto task : renderQueue_.grab())
        task();

    WGPUTextureFormat surfaceFormat = wgpuTextureGetFormat(target);

    if (!mainPipeline_)
        mainPipeline_ = createMainPipeline(device_, mainPipelineLayout_, surfaceFormat, shaderModule_);

    updateFrameBuffer({wgpuTextureGetWidth(target), wgpuTextureGetHeight(target)}, surfaceFormat);

    constexpr std::uint64_t minUniformBufferOffsetAlignment = 256;
    if (!objectUniformBuffer_ || wgpuBufferGetSize(objectUniformBuffer_) < objects.size() * minUniformBufferOffsetAlignment)
    {
        if (objectUniformBuffer_)
        {
            wgpuBindGroupRelease(objectBindGroup_);
            wgpuBufferRelease(objectUniformBuffer_);
        }

        objectUniformBuffer_ = createUniformBuffer(device_, objects.size() * minUniformBufferOffsetAlignment);

        objectBindGroup_ = createObjectBindGroup(device_, objectBindGroupLayout_, objectUniformBuffer_);

        for (auto const & object : objects)
            object->createTexturesBindGroup(device_, texturesBindGroupLayout_, defaultSampler_);
    }

    {
        struct ObjectUniformWrapper
        {
            ObjectUniform uniforms;
            char padding[minUniformBufferOffsetAlignment - sizeof(ObjectUniform)];
        };

        std::vector<ObjectUniformWrapper> objectUniforms(objects.size());
        for (int i = 0; i < objects.size(); ++i)
        {
            objectUniforms[i].uniforms = objects[i]->uniforms;
        }

        wgpuQueueWriteBuffer(queue_, objectUniformBuffer_, 0, objectUniforms.data(), objectUniforms.size() * sizeof(objectUniforms[0]));
    }

    WGPUTextureView targetView = createTextureView(target);
    WGPUCommandEncoder shadowCommandEncoder = createCommandEncoder(device_);

    glm::mat4 shadowProjection = computeShadowProjection(lightSettings.sunDirection, sceneBbox);

    updateCameraUniformBufferShadow(shadowProjection);

    WGPURenderPassEncoder shadowRenderPass = createShadowRenderPass(shadowCommandEncoder, shadowMapView_);

    wgpuRenderPassEncoderSetPipeline(shadowRenderPass, shadowPipeline_);
    wgpuRenderPassEncoderSetBindGroup(shadowRenderPass, 0, cameraBindGroup_, 0, nullptr);

    for (int i = 0; i < objects.size(); ++i)
    {
        auto const & object = objects[i];

        std::uint32_t dynamicOffset = i * minUniformBufferOffsetAlignment;

        wgpuRenderPassEncoderSetBindGroup(shadowRenderPass, 1, objectBindGroup_, 1, &dynamicOffset);
        wgpuRenderPassEncoderSetBindGroup(shadowRenderPass, 2, object->texturesBindGroup, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(shadowRenderPass, 0, object->common->vertexBuffer, object->vertexByteOffset, object->vertexByteLength);
        wgpuRenderPassEncoderSetIndexBuffer(shadowRenderPass, object->common->indexBuffer, object->indexFormat, object->indexByteOffset, object->indexByteLength);
        wgpuRenderPassEncoderDrawIndexed(shadowRenderPass, object->indexCount, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(shadowRenderPass);

    auto shadowCommandBuffer = commandEncoderFinish(shadowCommandEncoder);
    wgpuQueueSubmit(queue_, 1, &shadowCommandBuffer);

    WGPUCommandEncoder mainCommandEncoder = createCommandEncoder(device_);

    updateCameraUniformBuffer(camera);
    updateShadowUniformBuffer(shadowProjection, lightSettings);

    WGPURenderPassEncoder mainRenderPass = createMainRenderPass(mainCommandEncoder, frameTextureView_, depthTextureView_, targetView, glm::vec4(lightSettings.skyColor, 1.f));
    wgpuRenderPassEncoderSetPipeline(mainRenderPass, mainPipeline_);
    wgpuRenderPassEncoderSetBindGroup(mainRenderPass, 0, cameraBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(mainRenderPass, 3, lightsBindGroup_, 0, nullptr);

    for (int i = 0; i < objects.size(); ++i)
    {
        auto const & object = objects[i];

        std::uint32_t dynamicOffset = i * minUniformBufferOffsetAlignment;

        wgpuRenderPassEncoderSetBindGroup(mainRenderPass, 1, objectBindGroup_, 1, &dynamicOffset);
        wgpuRenderPassEncoderSetBindGroup(mainRenderPass, 2, object->texturesBindGroup, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(mainRenderPass, 0, object->common->vertexBuffer, object->vertexByteOffset, object->vertexByteLength);
        wgpuRenderPassEncoderSetIndexBuffer(mainRenderPass, object->common->indexBuffer, object->indexFormat, object->indexByteOffset, object->indexByteLength);
        wgpuRenderPassEncoderDrawIndexed(mainRenderPass, object->indexCount, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(mainRenderPass);

    auto mainCommandBuffer = commandEncoderFinish(mainCommandEncoder);
    wgpuQueueSubmit(queue_, 1, &mainCommandBuffer);

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

                renderObject->uniforms.modelMatrix = nodeModelMatrix;
                renderObject->uniforms.baseColorFactor = materialIn.baseColorFactor;
                renderObject->uniforms.metallicFactor = materialIn.metallicFactor;
                renderObject->uniforms.roughnessFactor = materialIn.roughnessFactor;
                renderObject->uniforms.emissiveFactor = materialIn.emissiveFactor;

                renderObject->textures.baseColorTextureId = materialIn.baseColorTexture;
                renderObject->textures.metallicRoughnessTextureId = materialIn.metallicRoughnessTexture;
                renderObject->textures.normalTextureId = materialIn.normalTexture;

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

                    auto transformedVertex = glm::vec3((renderObject->uniforms.modelMatrix * glm::vec4(vertices.back().position, 1.f)));

                    renderObject->bbox.expand(transformedVertex);
                }

                auto indexIterator = glTF::AccessorIterator<std::uint16_t>(assetBufferData.data() + indexBufferView.byteOffset + indexAccessor.byteOffset, indexBufferView.byteStride);
                for (int i = 0; i < indexAccessor.count; ++i)
                    indices.emplace_back(*indexIterator++);

                renderObject->createTexturesBindGroup(device_, texturesBindGroupLayout_, defaultSampler_);
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
        loaderQueue_.push([this, common, i]{ loadTexture(*common->textures[i]); });

    return result;
}

void Engine::Impl::updateFrameBuffer(glm::uvec2 const & renderTargetSize, WGPUTextureFormat surfaceFormat)
{
    if (!frameTexture_ || cachedRenderTargetSize_ != renderTargetSize)
    {
        if (frameTexture_)
        {
            wgpuTextureViewRelease(frameTextureView_);
            wgpuTextureRelease(frameTexture_);
            wgpuTextureViewRelease(depthTextureView_);
            wgpuTextureRelease(depthTexture_);
        }

        WGPUTextureDescriptor frameTextureDescriptor;
        frameTextureDescriptor.nextInChain = nullptr;
        frameTextureDescriptor.label = nullptr;
        frameTextureDescriptor.usage = WGPUTextureUsage_RenderAttachment;
        frameTextureDescriptor.dimension = WGPUTextureDimension_2D;
        frameTextureDescriptor.size = {renderTargetSize.x, renderTargetSize.y, 1};
        frameTextureDescriptor.format = surfaceFormat;
        frameTextureDescriptor.mipLevelCount = 1;
        frameTextureDescriptor.sampleCount = 4;
        frameTextureDescriptor.viewFormatCount = 0;
        frameTextureDescriptor.viewFormats = nullptr;

        frameTexture_ = wgpuDeviceCreateTexture(device_, &frameTextureDescriptor);
        frameTextureView_ = createTextureView(frameTexture_);

        WGPUTextureDescriptor depthTextureDescriptor;
        depthTextureDescriptor.nextInChain = nullptr;
        depthTextureDescriptor.label = nullptr;
        depthTextureDescriptor.usage = WGPUTextureUsage_RenderAttachment;
        depthTextureDescriptor.dimension = WGPUTextureDimension_2D;
        depthTextureDescriptor.size = {renderTargetSize.x, renderTargetSize.y, 1};
        depthTextureDescriptor.format = WGPUTextureFormat_Depth24Plus;
        depthTextureDescriptor.mipLevelCount = 1;
        depthTextureDescriptor.sampleCount = 4;
        depthTextureDescriptor.viewFormatCount = 0;
        depthTextureDescriptor.viewFormats = nullptr;

        depthTexture_ = wgpuDeviceCreateTexture(device_, &depthTextureDescriptor);
        depthTextureView_ = createTextureView(depthTexture_);

        cachedRenderTargetSize_ = renderTargetSize;
    }
}

void Engine::Impl::updateCameraUniformBuffer(Camera const & camera)
{
    CameraUniform cameraUniform;
    cameraUniform.viewProjection = glToVkProjection(camera.viewProjectionMatrix());
    cameraUniform.viewProjectionInverse = glm::inverse(cameraUniform.viewProjection);
    cameraUniform.position = camera.position();

    wgpuQueueWriteBuffer(queue_, cameraUniformBuffer_, 0, &cameraUniform, sizeof(CameraUniform));
}

glm::mat4 Engine::Impl::computeShadowProjection(glm::vec3 const & lightDirection, Box const & sceneBbox)
{
    glm::vec3 shadowZ = -glm::normalize(lightDirection);
    glm::vec3 shadowX = glm::normalize(glm::cross(shadowZ, glm::vec3(1.f, 0.f, 0.f)));
    glm::vec3 shadowY = glm::cross(shadowZ, shadowX);

    glm::vec3 sceneCenter = (sceneBbox.min + sceneBbox.max) / 2.f;
    glm::vec3 sceneHalfDiagonal = (sceneBbox.max - sceneBbox.min) / 2.f;

    glm::vec3 shadowExtent{0.f};

    shadowExtent[0] = std::abs(shadowX[0] * sceneHalfDiagonal[0]) + std::abs(shadowX[1] * sceneHalfDiagonal[1]) + std::abs(shadowX[2] * sceneHalfDiagonal[2]);
    shadowExtent[1] = std::abs(shadowY[0] * sceneHalfDiagonal[0]) + std::abs(shadowY[1] * sceneHalfDiagonal[1]) + std::abs(shadowY[2] * sceneHalfDiagonal[2]);
    shadowExtent[2] = std::abs(shadowZ[0] * sceneHalfDiagonal[0]) + std::abs(shadowZ[1] * sceneHalfDiagonal[1]) + std::abs(shadowZ[2] * sceneHalfDiagonal[2]);

    shadowX *= shadowExtent[0];
    shadowY *= shadowExtent[1];
    shadowZ *= shadowExtent[2];

    return glToVkProjection(glm::inverse(glm::mat4(
        glm::vec4(shadowX, 0.f),
        glm::vec4(shadowY, 0.f),
        glm::vec4(shadowZ, 0.f),
        glm::vec4(sceneCenter, 1.f)
    )));
}

void Engine::Impl::updateCameraUniformBufferShadow(glm::mat4 const & shadowProjection)
{
    CameraUniform cameraUniform;
    cameraUniform.viewProjection = shadowProjection;
    cameraUniform.viewProjectionInverse = glm::inverse(cameraUniform.viewProjection);
    cameraUniform.position = glm::vec3(0.f);

    wgpuQueueWriteBuffer(queue_, cameraUniformBuffer_, 0, &cameraUniform, sizeof(CameraUniform));
}

void Engine::Impl::updateShadowUniformBuffer(glm::mat4 const & shadowProjection, LightSettings const & lightSettings)
{
    LightsUniform shadowUniform;
    shadowUniform.shadowProjection = shadowProjection;
    shadowUniform.ambientLight = lightSettings.ambientLight;
    shadowUniform.sunDirection = lightSettings.sunDirection;
    shadowUniform.sunIntensity = lightSettings.sunIntensity;

    wgpuQueueWriteBuffer(queue_, lightsUniformBuffer_, 0, &shadowUniform, sizeof(LightsUniform));
}

void Engine::Impl::loadTexture(RenderObjectCommon::TextureInfo & textureInfo)
{
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

        // This should be possible to do in the loader thread once
        // wgpu-0.19 releases, see https://github.com/gfx-rs/wgpu/issues/4859
        renderQueue_.push([=, this, channels = imageInfo.channels]{
            WGPUImageCopyTexture imageCopyTexture;
            imageCopyTexture.nextInChain = nullptr;
            imageCopyTexture.texture = texture;
            imageCopyTexture.mipLevel = i;
            imageCopyTexture.origin = {0, 0, 0};
            imageCopyTexture.aspect = WGPUTextureAspect_All;

            WGPUTextureDataLayout textureDataLayout;
            textureDataLayout.nextInChain = nullptr;
            textureDataLayout.offset = 0;
            textureDataLayout.bytesPerRow = levelWidth * channels;
            textureDataLayout.rowsPerImage = levelHeight;

            WGPUExtent3D writeSize;
            writeSize.width = levelWidth;
            writeSize.height = levelHeight;
            writeSize.depthOrArrayLayers = 1;

            wgpuQueueWriteTexture(queue_, &imageCopyTexture, levelPixels.data(), levelPixels.size(), &textureDataLayout, &writeSize);
        });
    }

    textureInfo.texture.store(texture);

    for (auto const & user : textureInfo.users)
        renderQueue_.push([this, user]{
            if (auto object = user.lock())
                object->createTexturesBindGroup(device_, texturesBindGroupLayout_, defaultSampler_);
        });
}

void Engine::Impl::loaderThreadMain()
{
    while (true)
    {
        auto task = loaderQueue_.pop();
        if (!task)
            break;
        task();
    }
}

Engine::Engine(WGPUDevice device, WGPUQueue queue)
    : pimpl_(std::make_unique<Impl>(device, queue))
{}

Engine::~Engine() = default;

std::vector<RenderObjectPtr> Engine::loadGLTF(std::filesystem::path const & assetPath)
{
    return pimpl_->loadGLTF(assetPath);
}

void Engine::render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, LightSettings const & lightSettings)
{
    pimpl_->render(target, objects, camera, sceneBbox, lightSettings);
}

Box Engine::bbox(std::vector<RenderObjectPtr> const & objects) const
{
    Box result;

    for (auto const & object : objects)
        result.expand(object->bbox);

    return result;
}

