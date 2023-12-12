#include <webgpu-demo/engine_utils.hpp>

#include <vector>

glm::mat4 glToVkProjection(glm::mat4 matrix)
{
    // Map from [-1, 1] z-range in OpenGL to [0, 1] z-range in Vulkan, WebGPU, etc
    // Equivalent to doing v.z = (v.z + v.w) / 2 after applying the matrix

    for (int i = 0; i < 4; ++i)
        matrix[i][2] = (matrix[i][2] + matrix[i][3]) / 2.f;

    return matrix;
}

const char mainShader[] =
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
