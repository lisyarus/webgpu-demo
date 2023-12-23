#pragma once

#include <webgpu.h>

#include <glm/glm.hpp>

#include <initializer_list>
#include <filesystem>

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 texcoord;
    glm::vec4 rotation;
};

struct ClothVertex
{
    glm::vec3 oldVelocity;
    float padding1[1];
    glm::vec3 velocity;
    float padding2[1];
    glm::vec3 newPosition;
    float padding3[1];
};

struct ClothEdge
{
    glm::vec4 delta;
    std::uint32_t id;
    float padding1[3];
};

struct CameraUniform
{
    glm::mat4 viewProjection;
    glm::mat4 viewProjectionInverse;
    glm::vec3 position;
    float padding1[1];
    glm::vec4 shock;
    glm::vec3 shockDirection;
    float time;
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
    float envIntensity;
    std::uint32_t pointLightCount;
    float padding3[3];
};

struct PointLight
{
    glm::vec3 position;
    float padding1[1];
    glm::vec3 intensity;
    float padding2[1];
};

struct ClothSettingsUniform
{
    float dt;
    float gravity;
};

static constexpr std::size_t CLOTH_EDGES_PER_VERTEX = 8;

glm::mat4 glToVkProjection(glm::mat4 matrix);

extern const char mainShader[];
extern const char genMipmapShader[];
extern const char genEnvMipmapShader[];
extern const char blurShadowShader[];
extern const char simulateClothShader[];
extern const char ldrShader[];

std::uint32_t minStorageBufferOffsetAlignment(WGPUDevice device);

WGPUShaderModule createShaderModule(WGPUDevice device, char const * code);

WGPUBindGroupLayout createEmptyBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createCameraBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createObjectBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createTexturesBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createLightsBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createGenMipmapBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createGenEnvMipmapBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createBlurShadowBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createSimulateClothBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createHDRBindGroupLayout(WGPUDevice device);

WGPUPipelineLayout createPipelineLayout(WGPUDevice device, std::initializer_list<WGPUBindGroupLayout> bindGroupLayouts);

WGPURenderPipeline createMainPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPURenderPipeline createShadowPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPURenderPipeline createEnvPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createMipmapPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createMipmapSRGBPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createMipmapEnvPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createBlurShadowXPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createBlurShadowYPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createSimulateClothPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createSimulateClothCopyPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPURenderPipeline createRenderWaterPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPURenderPipeline createLDRPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule, WGPUTextureFormat surfaceFormat);

WGPUBuffer createUniformBuffer(WGPUDevice device, std::uint64_t size);
WGPUBuffer createStorageBuffer(WGPUDevice device, std::uint64_t size);

WGPUBindGroup createEmptyBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout);
WGPUBindGroup createCameraBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer);
WGPUBindGroup createObjectBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer);
WGPUBindGroup createLightsBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer,
    WGPUSampler shadowSampler, WGPUTextureView shadowMapView, WGPUSampler envSampler, WGPUTextureView envMapView,
    WGPUBuffer pointLightsBuffer, WGPUTextureView noise3DView, WGPUSampler noise3DSampler);
WGPUBindGroup createGenMipmapBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView input, WGPUTextureView output);
WGPUBindGroup createBlurShadowBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView input, WGPUTextureView output);
WGPUBindGroup createHDRBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView input);

WGPUTextureView createTextureView(WGPUTexture texture, int level = 0);
WGPUTextureView createTextureView(WGPUTexture texture, int level, WGPUTextureFormat format);

WGPUCommandEncoder createCommandEncoder(WGPUDevice device);

WGPURenderPassEncoder createMainRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget, WGPUTextureView resolveTarget, glm::vec4 const & clearColor);
WGPURenderPassEncoder createWaterRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget, WGPUTextureView resolveTarget, glm::vec4 const & clearColor);
WGPURenderPassEncoder createShadowRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget);
WGPURenderPassEncoder createEnvRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView resolveTarget);
WGPURenderPassEncoder createLDRRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView target);
WGPUComputePassEncoder createComputePass(WGPUCommandEncoder commandEncoder);

WGPUCommandBuffer commandEncoderFinish(WGPUCommandEncoder commandEncoder);

WGPUSampler createDefaultSampler(WGPUDevice device);
WGPUSampler createShadowSampler(WGPUDevice device);
WGPUSampler createEnvSampler(WGPUDevice device);
WGPUSampler create3DNoiseSampler(WGPUDevice device);

WGPUTexture createWhiteTexture(WGPUDevice device, WGPUQueue queue);
WGPUTexture createShadowMapTexture(WGPUDevice device, std::uint32_t size);
WGPUTexture createShadowMapDepthTexture(WGPUDevice device, std::uint32_t size);
WGPUTexture createStubEnvTexture(WGPUDevice device, WGPUQueue queue);
WGPUTexture create3DNoiseTexture(WGPUDevice device, WGPUQueue queue, std::filesystem::path const & path);
WGPUTextureView create3DNoiseTextureView(WGPUTexture texture);
