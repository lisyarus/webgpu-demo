#pragma once

#include <webgpu.h>

#include <glm/glm.hpp>

#include <initializer_list>
#include <vector>

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
    float envIntensity;
};

glm::mat4 glToVkProjection(glm::mat4 matrix);

extern const char mainShader[];
extern const char genMipmapShader[];
extern const char genEnvMipmapShader[];
extern const char summedAreaShadowShader[];

std::uint32_t getUniformBufferStride(WGPUDevice device);

WGPUShaderModule createShaderModule(WGPUDevice device, char const * code);

WGPUBindGroupLayout createEmptyBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createCameraBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createObjectBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createTexturesBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createLightsBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createGenMipmapBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createGenEnvMipmapBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createSummedAreaShadowTexturesBindGroupLayout(WGPUDevice device);
WGPUBindGroupLayout createSummedAreaShadowUniformsBindGroupLayout(WGPUDevice device);

WGPUPipelineLayout createPipelineLayout(WGPUDevice device, std::initializer_list<WGPUBindGroupLayout> bindGroupLayouts);

WGPURenderPipeline createMainPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUTextureFormat surfaceFormat, WGPUShaderModule shaderModule);
WGPURenderPipeline createShadowPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPURenderPipeline createEnvPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUTextureFormat surfaceFormat, WGPUShaderModule shaderModule);
WGPUComputePipeline createMipmapPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createMipmapSRGBPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createMipmapEnvPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createSummedAreaShadowPipelineX(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);
WGPUComputePipeline createSummedAreaShadowPipelineY(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule);

WGPUBuffer createUniformBuffer(WGPUDevice device, std::uint64_t size);
WGPUBuffer createSummedAreaShadowUniformBuffer(WGPUDevice device, WGPUQueue queue, WGPUTexture shadowMap, std::uint32_t uniformBufferStride);
int summedAreaShadowSteps(WGPUTexture shadowMap);

WGPUBindGroup createEmptyBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout);
WGPUBindGroup createCameraBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer);
WGPUBindGroup createObjectBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer);
WGPUBindGroup createLightsBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer,
    WGPUSampler shadowSampler, WGPUTextureView shadowMapView, WGPUSampler envSampler, WGPUTextureView envMapView);
WGPUBindGroup createGenMipmapBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView input, WGPUTextureView output);
WGPUBindGroup createSummedAreaShadowTexturesBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView shadowMap, WGPUTextureView shadowMapAux);
WGPUBindGroup createSummedAreaUniformsBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer);

WGPUTextureView createTextureView(WGPUTexture texture, int level = 0);
WGPUTextureView createTextureView(WGPUTexture texture, int level, WGPUTextureFormat format);

WGPUCommandEncoder createCommandEncoder(WGPUDevice device);

WGPURenderPassEncoder createMainRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget, WGPUTextureView resolveTarget, glm::vec4 const & clearColor);
WGPURenderPassEncoder createShadowRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget);
WGPURenderPassEncoder createEnvRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView resolveTarget);
WGPUComputePassEncoder createComputePass(WGPUCommandEncoder commandEncoder);

WGPUCommandBuffer commandEncoderFinish(WGPUCommandEncoder commandEncoder);

WGPUSampler createDefaultSampler(WGPUDevice device);
WGPUSampler createShadowSampler(WGPUDevice device);
WGPUSampler createEnvSampler(WGPUDevice device);

WGPUTexture createWhiteTexture(WGPUDevice device, WGPUQueue queue);
WGPUTexture createShadowMapTexture(WGPUDevice device, std::uint32_t size);
WGPUTexture createShadowMapDepthTexture(WGPUDevice device, std::uint32_t size);
WGPUTexture createStubEnvTexture(WGPUDevice device, WGPUQueue queue);
