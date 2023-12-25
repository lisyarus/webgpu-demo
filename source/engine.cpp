#include <webgpu-demo/engine.hpp>
#include <webgpu-demo/gltf_loader.hpp>
#include <webgpu-demo/gltf_iterator.hpp>
#include <webgpu-demo/synchronized_queue.hpp>
#include <webgpu-demo/engine_utils.hpp>
#include <webgpu-demo/render_object.hpp>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <stb_image.h>

#include <atomic>
#include <thread>
#include <functional>
#include <cstring>
#include <iostream>

struct Engine::Impl
{
    Impl(WGPUDevice device, WGPUQueue queue, std::filesystem::path const & noise3dPath, std::filesystem::path const & fontPath);
    ~Impl();

    void setEnvMap(std::filesystem::path const & hdrImagePath);
    void setWater(Box const & box, std::filesystem::path const & waterState);
    void render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, Settings const & settings, std::string const & text);
    std::vector<RenderObjectPtr> loadGLTF(std::filesystem::path const & assetPath);

private:
    WGPUDevice device_;
    WGPUQueue queue_;

    std::uint32_t minStorageBufferOffsetAlignment_;

    using Task = std::function<void()>;

    SynchronizedQueue<Task> loaderQueue_;
    SynchronizedQueue<Task> renderQueue_;
    std::thread loaderThread_;

    WGPUBindGroupLayout emptyBindGroupLayout_;
    WGPUBindGroupLayout cameraBindGroupLayout_;
    WGPUBindGroupLayout objectBindGroupLayout_;
    WGPUBindGroupLayout texturesBindGroupLayout_;
    WGPUBindGroupLayout lightsBindGroupLayout_;
    WGPUBindGroupLayout genMipmapBindGroupLayout_;
    WGPUBindGroupLayout genMipmapEnvBindGroupLayout_;
    WGPUBindGroupLayout blurShadowBindGroupLayout_;
    WGPUBindGroupLayout simulateClothBindGroupLayout_;
    WGPUBindGroupLayout waterBindGroupLayout_;
    WGPUBindGroupLayout simulateWaterBindGroupLayout_;
    WGPUBindGroupLayout hdrBindGroupLayout_;
    WGPUBindGroupLayout textBindGroupLayout_;

    WGPUShaderModule shaderModule_;
    WGPUShaderModule genMipmapShaderModule_;
    WGPUShaderModule genMipmapEnvShaderModule_;
    WGPUShaderModule blurShadowShaderModule_;
    WGPUShaderModule simulateClothShaderModule_;
    WGPUShaderModule simulateWaterShaderModule_;
    WGPUShaderModule ldrShaderModule_;
    WGPUShaderModule textShaderModule_;

    WGPUTexture shadowMap_;
    WGPUTexture shadowMapAux_;
    WGPUTexture shadowMapDepth_;
    WGPUTextureView shadowMapView_;
    WGPUTextureView shadowMapAuxView_;
    WGPUTextureView shadowMapDepthView_;

    WGPUSampler defaultSampler_;
    WGPUSampler shadowSampler_;
    WGPUSampler envSampler_;
    WGPUSampler noise3DSampler_;
    WGPUSampler textSampler_;

    WGPUPipelineLayout mainPipelineLayout_;
    WGPURenderPipeline mainPipeline_;
    WGPUPipelineLayout shadowPipelineLayout_;
    WGPURenderPipeline shadowPipeline_;
    WGPUPipelineLayout envPipelineLayout_;
    WGPURenderPipeline envPipeline_;
    WGPUPipelineLayout genMipmapPipelineLayout_;
    WGPUComputePipeline genMipmapPipeline_;
    WGPUComputePipeline genMipmapSRGBPipeline_;
    WGPUPipelineLayout genMipmapEnvPipelineLayout_;
    WGPUComputePipeline genMipmapEnvPipeline_;
    WGPUPipelineLayout blurShadowPipelineLayout_;
    WGPUComputePipeline blurShadowXPipeline_;
    WGPUComputePipeline blurShadowYPipeline_;
    WGPUPipelineLayout simulateClothPipelineLayout_;
    WGPUComputePipeline simulateClothPipeline_;
    WGPUComputePipeline simulateClothCopyPipeline_;
    WGPUPipelineLayout renderWaterPipelineLayout_;
    WGPURenderPipeline renderWaterPipeline_;
    WGPUPipelineLayout simulateWaterPipelineLayout_;
    WGPUComputePipeline simulateWaterPipeline_;
    WGPUPipelineLayout ldrPipelineLayout_;
    WGPURenderPipeline ldrPipeline_;
    WGPUPipelineLayout textPipelineLayout_;
    WGPURenderPipeline textPipeline_;

    WGPUBuffer cameraUniformBuffer_;
    WGPUBuffer objectUniformBuffer_;
    WGPUBuffer lightsUniformBuffer_;
    WGPUBuffer clothSettingsUniformBuffer_;
    WGPUBuffer pointLightsBuffer_;
    WGPUBuffer waterUniformBuffer_;
    WGPUBuffer textUniformBuffer_;
    WGPUBuffer textVertexBuffer_;

    WGPUTexture stubEnvTexture_;
    WGPUTexture envTexture_;
    WGPUTextureView envTextureView_;
    WGPUTexture noise3DTexture_;
    WGPUTextureView noise3DTextureView_;
    WGPUTexture fontTexture_;
    WGPUTextureView fontTextureView_;

    std::uint64_t objectUniformBufferStride_ = 256;

    WGPUBindGroup emptyBindGroup_;
    WGPUBindGroup cameraBindGroup_;
    WGPUBindGroup objectBindGroup_;
    WGPUBindGroup lightsBindGroup_;
    WGPUBindGroup blurShadowXBindGroup_;
    WGPUBindGroup blurShadowYBindGroup_;
    WGPUBindGroup waterBindGroup_;
    WGPUBindGroup simulateWaterBindGroup1_;
    WGPUBindGroup simulateWaterBindGroup2_;
    WGPUBindGroup hdrBindGroup_;
    WGPUBindGroup textBindGroup_;

    WGPUTexture hdrMultisampleTexture_;
    WGPUTextureView hdrMultisampleTextureView_;
    WGPUTexture hdrResolveTexture_;
    WGPUTextureView hdrResolveTextureView_;
    WGPUTexture hdrWaterTexture_;
    WGPUTextureView hdrWaterTextureView_;
    WGPUTexture multisampleDepthTexture_;
    WGPUTextureView multisampleDepthTextureView_;

    WGPUTexture whiteTexture_;

    Box waterBbox_;
    glm::ivec2 waterCellCount_;
    glm::vec2 waterCellSize_;
    WGPUBuffer waterGridVertexBuffer_;
    WGPUBuffer waterGridIndexBuffer_;
    WGPUTexture waterDataTexture1_;
    WGPUTexture waterDataTexture2_;
    WGPUTextureView waterDataTextureView1_;
    WGPUTextureView waterDataTextureView2_;

    glm::uvec2 cachedRenderTargetSize_{0, 0};

    void simulateCloth(std::vector<RenderObjectPtr> const & objects, Camera const & camera, Settings const & settings, int iterations);
    void renderShadow(std::vector<RenderObjectPtr> const & objects);
    void blurShadow();
    void renderEnv();
    void renderMain(std::vector<RenderObjectPtr> const & objects);
    void renderWater(Settings const & settings);
    void renderLDR(WGPUTextureView targetView, WGPUTextureFormat surfaceFormat);
    void renderText(WGPUTextureView targetView, WGPUTextureFormat surfaceFormat, glm::ivec2 const & viewportSize, std::string const & text);

    void updateFrameBuffer(glm::uvec2 const & renderTargetSize);
    void updateCameraBuffer(Camera const & camera, Settings const & settings, WGPUBuffer buffer, glm::ivec2 const & viewport);
    void updateObjectUniformBuffer(std::vector<RenderObjectPtr> const & objects);
    int updatePointLightsBuffer(std::vector<RenderObjectPtr> const & objects);
    glm::mat4 computeShadowProjection(glm::vec3 const & lightDirection, Box const & sceneBbox);
    void updateCameraUniformBufferShadow(glm::mat4 const & shadowProjection);
    void updateLightsUniformBuffer(glm::mat4 const & shadowProjection, Settings const & settings, int pointLightCount);
    void loadTexture(RenderObjectCommon::TextureInfo & textureInfo);
    void recreateLightsBindGroup();
    void loaderThreadMain();
};

Engine::Impl::Impl(WGPUDevice device, WGPUQueue queue, std::filesystem::path const & noise3DPath, std::filesystem::path const & fontPath)
    : device_(device)
    , queue_(queue)
    , minStorageBufferOffsetAlignment_(minStorageBufferOffsetAlignment(device_))
    , loaderThread_([this]{ loaderThreadMain(); })

    // Bind group layouts
    , emptyBindGroupLayout_(createEmptyBindGroupLayout(device_))
    , cameraBindGroupLayout_(createCameraBindGroupLayout(device_))
    , objectBindGroupLayout_(createObjectBindGroupLayout(device_))
    , texturesBindGroupLayout_(createTexturesBindGroupLayout(device_))
    , lightsBindGroupLayout_(createLightsBindGroupLayout(device_))
    , genMipmapBindGroupLayout_(createGenMipmapBindGroupLayout(device_))
    , genMipmapEnvBindGroupLayout_(createGenEnvMipmapBindGroupLayout(device_))
    , blurShadowBindGroupLayout_(createBlurShadowBindGroupLayout(device_))
    , simulateClothBindGroupLayout_(createSimulateClothBindGroupLayout(device_))
    , waterBindGroupLayout_(createWaterBindGroupLayout(device_))
    , simulateWaterBindGroupLayout_(createSimulateWaterBindGroupLayout(device_))
    , hdrBindGroupLayout_(createHDRBindGroupLayout(device_))
    , textBindGroupLayout_(createTextBindGroupLayout(device_))

    // Shader modules
    , shaderModule_(createShaderModule(device_, mainShader))
    , genMipmapShaderModule_(createShaderModule(device_, genMipmapShader))
    , genMipmapEnvShaderModule_(createShaderModule(device_, genEnvMipmapShader))
    , blurShadowShaderModule_(createShaderModule(device_, blurShadowShader))
    , simulateClothShaderModule_(createShaderModule(device_, simulateClothShader))
    , simulateWaterShaderModule_(createShaderModule(device_, simulateWaterShader))
    , ldrShaderModule_(createShaderModule(device_, ldrShader))
    , textShaderModule_(createShaderModule(device_, textShader))

    // Shadow map textures & views
    , shadowMap_(createShadowMapTexture(device_, 2048))
    , shadowMapAux_(createShadowMapTexture(device_, 2048))
    , shadowMapDepth_(createShadowMapDepthTexture(device_, 2048))
    , shadowMapView_(createTextureView(shadowMap_))
    , shadowMapAuxView_(createTextureView(shadowMapAux_))
    , shadowMapDepthView_(createTextureView(shadowMapDepth_))

    // Samplers
    , defaultSampler_(createDefaultSampler(device_))
    , shadowSampler_(createShadowSampler(device_))
    , envSampler_(createEnvSampler(device_))
    , noise3DSampler_(create3DNoiseSampler(device_))
    , textSampler_(createTextSampler(device_))

    // Pipelines
    , mainPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, objectBindGroupLayout_, texturesBindGroupLayout_, lightsBindGroupLayout_}))
    , mainPipeline_(createMainPipeline(device_, mainPipelineLayout_, shaderModule_))
    , shadowPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, objectBindGroupLayout_, texturesBindGroupLayout_}))
    , shadowPipeline_(createShadowPipeline(device_, shadowPipelineLayout_, shaderModule_))
    , envPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, emptyBindGroupLayout_, emptyBindGroupLayout_, lightsBindGroupLayout_}))
    , envPipeline_(createEnvPipeline(device_, envPipelineLayout_, shaderModule_))
    , genMipmapPipelineLayout_(createPipelineLayout(device_, {genMipmapBindGroupLayout_}))
    , genMipmapPipeline_(createMipmapPipeline(device_, genMipmapPipelineLayout_, genMipmapShaderModule_))
    , genMipmapSRGBPipeline_(createMipmapSRGBPipeline(device_, genMipmapPipelineLayout_, genMipmapShaderModule_))
    , genMipmapEnvPipelineLayout_(createPipelineLayout(device_, {genMipmapEnvBindGroupLayout_}))
    , genMipmapEnvPipeline_(createMipmapEnvPipeline(device_, genMipmapEnvPipelineLayout_, genMipmapEnvShaderModule_))
    , blurShadowPipelineLayout_(createPipelineLayout(device_, {blurShadowBindGroupLayout_}))
    , blurShadowXPipeline_(createBlurShadowXPipeline(device_, blurShadowPipelineLayout_, blurShadowShaderModule_))
    , blurShadowYPipeline_(createBlurShadowYPipeline(device_, blurShadowPipelineLayout_, blurShadowShaderModule_))
    , simulateClothPipelineLayout_(createPipelineLayout(device_, {simulateClothBindGroupLayout_, cameraBindGroupLayout_}))
    , simulateClothPipeline_(createSimulateClothPipeline(device_, simulateClothPipelineLayout_, simulateClothShaderModule_))
    , simulateClothCopyPipeline_(createSimulateClothCopyPipeline(device_, simulateClothPipelineLayout_, simulateClothShaderModule_))
    , renderWaterPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, emptyBindGroupLayout_, emptyBindGroupLayout_, lightsBindGroupLayout_, waterBindGroupLayout_}))
    , renderWaterPipeline_(createRenderWaterPipeline(device_, renderWaterPipelineLayout_, shaderModule_))
    , simulateWaterPipelineLayout_(createPipelineLayout(device_, {simulateWaterBindGroupLayout_}))
    , simulateWaterPipeline_(createSimulateWaterPipeline(device_, simulateWaterPipelineLayout_, simulateWaterShaderModule_))
    , ldrPipelineLayout_(createPipelineLayout(device_, {hdrBindGroupLayout_}))
    , ldrPipeline_(nullptr)
    , textPipelineLayout_(createPipelineLayout(device_, {textBindGroupLayout_}))
    , textPipeline_(nullptr)

    // Uniform buffers
    , cameraUniformBuffer_(createUniformBuffer(device_, sizeof(CameraUniform)))
    , objectUniformBuffer_(nullptr)
    , lightsUniformBuffer_(createUniformBuffer(device_, sizeof(LightsUniform)))
    , clothSettingsUniformBuffer_(createUniformBuffer(device_, sizeof(ClothSettingsUniform)))
    , pointLightsBuffer_(createStorageBuffer(device_, 16 * sizeof(PointLight)))
    , waterUniformBuffer_(createUniformBuffer(device_, sizeof(WaterUniform)))
    , textUniformBuffer_(createUniformBuffer(device_, sizeof(TextUniform)))
    , textVertexBuffer_(nullptr)

    // Environment map textures and views
    , stubEnvTexture_(createStubEnvTexture(device_, queue_))
    , envTexture_(nullptr)
    , envTextureView_(createTextureView(stubEnvTexture_))

    // Noise textures
    , noise3DTexture_(create3DNoiseTexture(device_, queue_, noise3DPath))
    , noise3DTextureView_(create3DNoiseTextureView(noise3DTexture_))

    // Font textures
    , fontTexture_(createFontTexture(device_, queue_, fontPath))
    , fontTextureView_(createTextureView(fontTexture_))

    // Bind groups
    , emptyBindGroup_(createEmptyBindGroup(device_, emptyBindGroupLayout_))
    , cameraBindGroup_(createCameraBindGroup(device_, cameraBindGroupLayout_, cameraUniformBuffer_))
    , objectBindGroup_(nullptr)
    , lightsBindGroup_(nullptr)
    , blurShadowXBindGroup_(createBlurShadowBindGroup(device_, blurShadowBindGroupLayout_, shadowMapView_, shadowMapAuxView_))
    , blurShadowYBindGroup_(createBlurShadowBindGroup(device_, blurShadowBindGroupLayout_, shadowMapAuxView_, shadowMapView_))
    , waterBindGroup_(nullptr)
    , simulateWaterBindGroup1_(nullptr)
    , simulateWaterBindGroup2_(nullptr)
    , hdrBindGroup_(nullptr)
    , textBindGroup_(createTextBindGroup(device_, textBindGroupLayout_, textUniformBuffer_, fontTextureView_, textSampler_))

    // Frame textures
    , hdrMultisampleTexture_(nullptr)
    , hdrMultisampleTextureView_(nullptr)
    , hdrResolveTexture_(nullptr)
    , hdrResolveTextureView_(nullptr)
    , hdrWaterTexture_(nullptr)
    , hdrWaterTextureView_(nullptr)
    , multisampleDepthTexture_(nullptr)
    , multisampleDepthTextureView_(nullptr)

    // Extra textures
    , whiteTexture_(createWhiteTexture(device_, queue_))

    // Water
    , waterBbox_{{0.f, 0.f, 0.f}, {0.f, 0.f, 1.f}}
    , waterCellCount_{0, 0}
    , waterGridVertexBuffer_(nullptr)
    , waterGridIndexBuffer_(nullptr)
    , waterDataTexture1_(nullptr)
    , waterDataTexture2_(nullptr)
    , waterDataTextureView1_(nullptr)
    , waterDataTextureView2_(nullptr)
{
    recreateLightsBindGroup();
}

Engine::Impl::~Impl()
{
    loaderQueue_.grab();
    loaderQueue_.push(nullptr);
    loaderThread_.join();

    if (emptyBindGroupLayout_) wgpuBindGroupLayoutRelease(emptyBindGroupLayout_);
    if (cameraBindGroupLayout_) wgpuBindGroupLayoutRelease(cameraBindGroupLayout_);
    if (objectBindGroupLayout_) wgpuBindGroupLayoutRelease(objectBindGroupLayout_);
    if (texturesBindGroupLayout_) wgpuBindGroupLayoutRelease(texturesBindGroupLayout_);
    if (lightsBindGroupLayout_) wgpuBindGroupLayoutRelease(lightsBindGroupLayout_);
    if (genMipmapBindGroupLayout_) wgpuBindGroupLayoutRelease(genMipmapBindGroupLayout_);
    if (genMipmapEnvBindGroupLayout_) wgpuBindGroupLayoutRelease(genMipmapEnvBindGroupLayout_);
    if (blurShadowBindGroupLayout_) wgpuBindGroupLayoutRelease(blurShadowBindGroupLayout_);
    if (simulateClothBindGroupLayout_) wgpuBindGroupLayoutRelease(simulateClothBindGroupLayout_);

    if (shaderModule_) wgpuShaderModuleRelease(shaderModule_);
    if (genMipmapShaderModule_) wgpuShaderModuleRelease(genMipmapShaderModule_);
    if (genMipmapEnvShaderModule_) wgpuShaderModuleRelease(genMipmapEnvShaderModule_);
    if (blurShadowShaderModule_) wgpuShaderModuleRelease(blurShadowShaderModule_);
    if (simulateClothShaderModule_) wgpuShaderModuleRelease(simulateClothShaderModule_);

    if (shadowMap_) wgpuTextureRelease(shadowMap_);
    if (shadowMapAux_) wgpuTextureRelease(shadowMapAux_);
    if (shadowMapDepth_) wgpuTextureRelease(shadowMapDepth_);
    if (shadowMapView_) wgpuTextureViewRelease(shadowMapView_);
    if (shadowMapAuxView_) wgpuTextureViewRelease(shadowMapAuxView_);
    if (shadowMapDepthView_) wgpuTextureViewRelease(shadowMapDepthView_);

    if (defaultSampler_) wgpuSamplerRelease(defaultSampler_);
    if (shadowSampler_) wgpuSamplerRelease(shadowSampler_);
    if (envSampler_) wgpuSamplerRelease(envSampler_);

    if (mainPipelineLayout_) wgpuPipelineLayoutRelease(mainPipelineLayout_);
    if (mainPipeline_) wgpuRenderPipelineRelease(mainPipeline_);
    if (shadowPipelineLayout_) wgpuPipelineLayoutRelease(shadowPipelineLayout_);
    if (shadowPipeline_) wgpuRenderPipelineRelease(shadowPipeline_);
    if (envPipelineLayout_) wgpuPipelineLayoutRelease(envPipelineLayout_);
    if (envPipeline_) wgpuRenderPipelineRelease(envPipeline_);
    if (genMipmapPipelineLayout_) wgpuPipelineLayoutRelease(genMipmapPipelineLayout_);
    if (genMipmapPipeline_) wgpuComputePipelineRelease(genMipmapPipeline_);
    if (genMipmapSRGBPipeline_) wgpuComputePipelineRelease(genMipmapSRGBPipeline_);
    if (genMipmapEnvPipelineLayout_) wgpuPipelineLayoutRelease(genMipmapEnvPipelineLayout_);
    if (genMipmapEnvPipeline_) wgpuComputePipelineRelease(genMipmapEnvPipeline_);
    if (blurShadowPipelineLayout_) wgpuPipelineLayoutRelease(blurShadowPipelineLayout_);
    if (blurShadowXPipeline_) wgpuComputePipelineRelease(blurShadowXPipeline_);
    if (blurShadowYPipeline_) wgpuComputePipelineRelease(blurShadowYPipeline_);
    if (simulateClothPipelineLayout_) wgpuPipelineLayoutRelease(simulateClothPipelineLayout_);
    if (simulateClothPipeline_) wgpuComputePipelineRelease(simulateClothPipeline_);
    if (simulateClothCopyPipeline_) wgpuComputePipelineRelease(simulateClothCopyPipeline_);

    if (cameraUniformBuffer_) wgpuBufferRelease(cameraUniformBuffer_);
    if (objectUniformBuffer_) wgpuBufferRelease(objectUniformBuffer_);
    if (lightsUniformBuffer_) wgpuBufferRelease(lightsUniformBuffer_);
    if (clothSettingsUniformBuffer_) wgpuBufferRelease(clothSettingsUniformBuffer_);
    if (pointLightsBuffer_) wgpuBufferRelease(pointLightsBuffer_);

    if (stubEnvTexture_) wgpuTextureRelease(stubEnvTexture_);
    if (envTexture_) wgpuTextureRelease(envTexture_);
    if (envTextureView_) wgpuTextureViewRelease(envTextureView_);

    if (emptyBindGroup_) wgpuBindGroupRelease(emptyBindGroup_);
    if (cameraBindGroup_) wgpuBindGroupRelease(cameraBindGroup_);
    if (objectBindGroup_) wgpuBindGroupRelease(objectBindGroup_);
    if (lightsBindGroup_) wgpuBindGroupRelease(lightsBindGroup_);
    if (blurShadowXBindGroup_) wgpuBindGroupRelease(blurShadowXBindGroup_);
    if (blurShadowYBindGroup_) wgpuBindGroupRelease(blurShadowYBindGroup_);

    if (hdrMultisampleTexture_) wgpuTextureRelease(hdrMultisampleTexture_);
    if (hdrMultisampleTextureView_) wgpuTextureViewRelease(hdrMultisampleTextureView_);
    if (multisampleDepthTexture_) wgpuTextureRelease(multisampleDepthTexture_);
    if (multisampleDepthTextureView_) wgpuTextureViewRelease(multisampleDepthTextureView_);

    if (whiteTexture_) wgpuTextureRelease(whiteTexture_);

}

void Engine::Impl::setEnvMap(std::filesystem::path const & hdrImagePath)
{
    loaderQueue_.push([this, hdrImagePath]{
        int width, height, channels;
        float * pixels = stbi_loadf(hdrImagePath.c_str(), &width, &height, &channels, 4);

        WGPUTextureDescriptor textureDescriptor;
        textureDescriptor.nextInChain = nullptr;
        textureDescriptor.label = nullptr;
        textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding;
        textureDescriptor.dimension = WGPUTextureDimension_2D;
        textureDescriptor.size = {(std::uint32_t)width, (std::uint32_t)height, 1};
        textureDescriptor.format = WGPUTextureFormat_RGBA32Float;
        textureDescriptor.mipLevelCount = std::floor(std::log2(std::max(width, height))) + 1;
        textureDescriptor.sampleCount = 1;
        textureDescriptor.viewFormatCount = 0;
        textureDescriptor.viewFormats = nullptr;

        WGPUTexture texture = wgpuDeviceCreateTexture(device_, &textureDescriptor);

        WGPUImageCopyTexture imageCopyTexture;
        imageCopyTexture.nextInChain = nullptr;
        imageCopyTexture.texture = texture;
        imageCopyTexture.mipLevel = 0;
        imageCopyTexture.origin = {0, 0, 0};
        imageCopyTexture.aspect = WGPUTextureAspect_All;

        WGPUTextureDataLayout textureDataLayout;
        textureDataLayout.nextInChain = nullptr;
        textureDataLayout.offset = 0;
        textureDataLayout.bytesPerRow = width * 4 * sizeof(float);
        textureDataLayout.rowsPerImage = height;

        WGPUExtent3D writeSize;
        writeSize.width = width;
        writeSize.height = height;
        writeSize.depthOrArrayLayers = 1;

        wgpuQueueWriteTexture(queue_, &imageCopyTexture, pixels, width * height * 4 * sizeof(float), &textureDataLayout, &writeSize);

        stbi_image_free(pixels);

        std::vector<WGPUBindGroup> bindGroups;

        WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);
        WGPUComputePassEncoder computePass = createComputePass(commandEncoder);
        wgpuComputePassEncoderSetPipeline(computePass, genMipmapEnvPipeline_);

        for (int level = 1; level < textureDescriptor.mipLevelCount; ++level)
        {
            auto inputView = createTextureView(texture, level - 1);
            auto outputView = createTextureView(texture, level);
            auto bindGroup = createGenMipmapBindGroup(device_, genMipmapEnvBindGroupLayout_, inputView, outputView);

            int workgroupCountX = std::ceil((width >> level) / 8.f);
            int workgroupCountY = std::ceil((height >> level) / 8.f);

            wgpuComputePassEncoderSetBindGroup(computePass, 0, bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(computePass, workgroupCountX, workgroupCountY, 1);

            bindGroups.push_back(bindGroup);
            wgpuTextureViewRelease(outputView);
            wgpuTextureViewRelease(inputView);
        }

        wgpuComputePassEncoderEnd(computePass);
        wgpuComputePassEncoderRelease(computePass);

        // Bind groups should live as long as the compute pass lives,
        // so we defer it's release until the compute pass ends
        for (auto bindGroup : bindGroups)
            wgpuBindGroupRelease(bindGroup);

        WGPUCommandBuffer commandBuffer = commandEncoderFinish(commandEncoder);

        wgpuCommandEncoderRelease(commandEncoder);

        // This is probably a bug in wgpu, but submitting a compute pass
        // in a separate thread causes GPU hangs or device loss, see
        // https://github.com/gfx-rs/wgpu/issues/4877
        renderQueue_.push([this, commandBuffer]
        {
            wgpuQueueSubmit(queue_, 1, &commandBuffer);
            wgpuCommandBufferRelease(commandBuffer);
        });

        WGPUTextureViewDescriptor textureViewDescriptor;
        textureViewDescriptor.nextInChain = nullptr;
        textureViewDescriptor.label = nullptr;
        textureViewDescriptor.format = WGPUTextureFormat_RGBA32Float;
        textureViewDescriptor.dimension = WGPUTextureViewDimension_2D;
        textureViewDescriptor.baseMipLevel = 0;
        textureViewDescriptor.mipLevelCount = textureDescriptor.mipLevelCount;
        textureViewDescriptor.baseArrayLayer = 0;
        textureViewDescriptor.arrayLayerCount = 1;
        textureViewDescriptor.aspect = WGPUTextureAspect_All;

        WGPUTextureView textureView = wgpuTextureCreateView(texture, &textureViewDescriptor);

        renderQueue_.push([this, texture, textureView]{
            if (envTexture_)
                wgpuTextureRelease(envTexture_);

            envTexture_ = texture;
            envTextureView_ = textureView;

            recreateLightsBindGroup();
        });
    });
}

void Engine::Impl::setWater(Box const & box, std::filesystem::path const & waterState)
{
    waterBbox_ = box;

    auto const boxSize = box.diagonal();

    float const maxDimension = std::max(boxSize.x, boxSize.y);

    int const maxDimensionCellCount = 1024;

    waterCellCount_ =
    {
        std::round((boxSize.x * maxDimensionCellCount) / maxDimension),
        std::round((boxSize.z * maxDimensionCellCount) / maxDimension)
    };

    waterCellCount_.x += (8 - (waterCellCount_.x % 8)) % 8;
    waterCellCount_.y += (8 - (waterCellCount_.y % 8)) % 8;

    waterCellSize_ =
    {
        boxSize.x / (waterCellCount_.x - 1),
        boxSize.z / (waterCellCount_.y - 1),
    };

    std::vector<glm::vec2> gridVertices;
    std::vector<std::uint32_t> gridIndices;
    gridVertices.reserve(waterCellCount_.x * waterCellCount_.y);
    gridIndices.reserve(6 * (waterCellCount_.x - 1) * (waterCellCount_.y - 1));

    for (int y = 0; y < waterCellCount_.y; ++y)
        for (int x = 0; x < waterCellCount_.x; ++x)
            gridVertices.push_back(glm::vec2(box.min.x, box.min.z) + glm::vec2(x, y) * waterCellSize_);

    for (int y = 0; y + 1 < waterCellCount_.y; ++y)
    {
        for (int x = 0; x + 1 < waterCellCount_.x; ++x)
        {
            std::uint32_t i00 = (y + 0) * waterCellCount_.x + x + 0;
            std::uint32_t i01 = (y + 0) * waterCellCount_.x + x + 1;
            std::uint32_t i10 = (y + 1) * waterCellCount_.x + x + 0;
            std::uint32_t i11 = (y + 1) * waterCellCount_.x + x + 1;

            gridIndices.insert(gridIndices.end(), {i00, i01, i10, i10, i01, i11});
        }
    }

    if (waterGridVertexBuffer_)
        wgpuBufferRelease(waterGridVertexBuffer_);

    if (waterGridIndexBuffer_)
        wgpuBufferRelease(waterGridIndexBuffer_);

    {
        WGPUBufferDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
        descriptor.size = gridVertices.size() * sizeof(gridVertices[0]);
        descriptor.mappedAtCreation = false;

        waterGridVertexBuffer_ = wgpuDeviceCreateBuffer(device_, &descriptor);
        wgpuQueueWriteBuffer(queue_, waterGridVertexBuffer_, 0, gridVertices.data(), gridVertices.size() * sizeof(gridVertices[0]));
    }

    {
        WGPUBufferDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index;
        descriptor.size = gridIndices.size() * sizeof(gridIndices[0]);
        descriptor.mappedAtCreation = false;

        waterGridIndexBuffer_ = wgpuDeviceCreateBuffer(device_, &descriptor);
        wgpuQueueWriteBuffer(queue_, waterGridIndexBuffer_, 0, gridIndices.data(), gridIndices.size() * sizeof(gridIndices[0]));
    }

    if (waterDataTexture1_)
        wgpuTextureRelease(waterDataTexture1_);

    if (waterDataTexture2_)
        wgpuTextureRelease(waterDataTexture2_);

    if (waterDataTextureView1_)
        wgpuTextureViewRelease(waterDataTextureView1_);

    if (waterDataTextureView2_)
        wgpuTextureViewRelease(waterDataTextureView2_);

    {
        WGPUTextureDescriptor descriptor;
        descriptor.nextInChain = nullptr;
        descriptor.label = nullptr;
        descriptor.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        descriptor.dimension = WGPUTextureDimension_2D;
        descriptor.size = {(std::uint32_t)waterCellCount_.x, (std::uint32_t)waterCellCount_.y, 1};
        descriptor.format = WGPUTextureFormat_RG32Float;
        descriptor.mipLevelCount = 1;
        descriptor.sampleCount = 1;
        descriptor.viewFormatCount = 0;
        descriptor.viewFormats = nullptr;

        waterDataTexture1_ = wgpuDeviceCreateTexture(device_, &descriptor);
        waterDataTexture2_ = wgpuDeviceCreateTexture(device_, &descriptor);

        waterDataTextureView1_ = createTextureView(waterDataTexture1_);
        waterDataTextureView2_ = createTextureView(waterDataTexture2_);

        auto image = glTF::loadImage(waterState);

        std::vector<float> pixels;
        for (int y = 0; y < waterCellCount_.y; ++y)
        {
            for (int x = 0; x < waterCellCount_.x; ++x)
            {
                pixels.push_back(image.data.get()[y * image.width + x] / 255.f - 0.5f);
                pixels.push_back(0.0f);
            }
        }

        WGPUImageCopyTexture destination;
        destination.nextInChain = nullptr;
        destination.texture = waterDataTexture1_;
        destination.mipLevel = 0;
        destination.origin = {0, 0, 0};
        destination.aspect = WGPUTextureAspect_All;

        WGPUExtent3D writeSize;
        writeSize.width = waterCellCount_.x;
        writeSize.height = waterCellCount_.y;
        writeSize.depthOrArrayLayers = 1;

        WGPUTextureDataLayout dataLayout;
        dataLayout.nextInChain = nullptr;
        dataLayout.offset = 0;
        dataLayout.bytesPerRow = writeSize.width * sizeof(pixels[0]) * 2;
        dataLayout.rowsPerImage = writeSize.height;

        wgpuQueueWriteTexture(queue_, &destination, pixels.data(), pixels.size() * sizeof(pixels[0]), &dataLayout, &writeSize);
    }

    if (waterBindGroup_)
        wgpuBindGroupRelease(waterBindGroup_);
    waterBindGroup_ = nullptr;

    if (simulateWaterBindGroup1_)
        wgpuBindGroupRelease(simulateWaterBindGroup1_);
    simulateWaterBindGroup1_ = nullptr;

    if (simulateWaterBindGroup2_)
        wgpuBindGroupRelease(simulateWaterBindGroup2_);
    simulateWaterBindGroup2_ = nullptr;
}

void Engine::Impl::render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, Settings const & settings, std::string const & text)
{
    for (auto task : renderQueue_.grab())
        task();

    updateFrameBuffer({wgpuTextureGetWidth(target), wgpuTextureGetHeight(target)});

    updateObjectUniformBuffer(objects);

    int pointLightCount = updatePointLightsBuffer(objects);

    glm::ivec2 const viewportSize{wgpuTextureGetWidth(target), wgpuTextureGetHeight(target)};

    updateCameraBuffer(camera, settings, cameraUniformBuffer_, viewportSize);
    simulateCloth(objects, camera, settings, settings.paused ? 0 : 16);

    glm::mat4 shadowProjection = computeShadowProjection(settings.sunDirection, sceneBbox);

    updateCameraUniformBufferShadow(shadowProjection);
    renderShadow(objects);

    blurShadow();

    updateCameraBuffer(camera, settings, cameraUniformBuffer_, viewportSize);
    updateLightsUniformBuffer(shadowProjection, settings, pointLightCount);

    renderEnv();
    renderMain(objects);
    renderWater(settings);

    WGPUTextureFormat surfaceFormat = wgpuTextureGetFormat(target);
    WGPUTextureView targetView = createTextureView(target);

    renderLDR(targetView, surfaceFormat);

    renderText(targetView, surfaceFormat, viewportSize, text);

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
        if (asset.buffers.size() > 1)
            throw std::runtime_error("Only 1 binary buffer is supported");

        std::vector<char> assetBufferData;
        if (!asset.buffers.empty())
            assetBufferData = glTF::loadBuffer(assetPath, asset.buffers[0].uri);

        std::vector<Vertex> vertices;
        std::vector<std::uint16_t> indices;
        std::vector<ClothVertex> clothVertices;
        std::vector<ClothEdge> clothEdges;

        for (auto const & node : asset.nodes)
        {
            glm::mat4 nodeModelMatrix =
                glm::scale(node.scale) *
                glm::toMat4(node.rotation) *
                glm::translate(node.translation);

            if (node.light)
            {
                auto & renderObject = result.emplace_back(std::make_shared<RenderObject>());

                renderObject->common = nullptr;

                renderObject->vertices.byteOffset = 0;
                renderObject->vertices.byteLength = 0;
                renderObject->vertices.count = 0;
                renderObject->indices.byteOffset = 0;
                renderObject->indices.byteLength = 0;
                renderObject->indices.count = 0;
                renderObject->indexFormat = WGPUIndexFormat_Undefined;

                renderObject->light = RenderObject::Light{node.translation, asset.lights[*node.light].intensity};
            }

            if (node.mesh)
            {
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

                    renderObject->vertices.byteOffset = vertices.size() * sizeof(vertices[0]);
                    renderObject->vertices.byteLength = positionAccessor.count * sizeof(vertices[0]);
                    renderObject->vertices.count = positionAccessor.count;
                    renderObject->indices.byteOffset = indices.size() * sizeof(indices[0]);
                    renderObject->indices.byteLength = indexAccessor.count * sizeof(indices[0]);
                    renderObject->indices.count = indexAccessor.count;
                    renderObject->indexFormat = WGPUIndexFormat_Uint16;

                    renderObject->uniforms.modelMatrix = glm::mat4(1.f);
                    renderObject->uniforms.baseColorFactor = materialIn.baseColorFactor;
                    renderObject->uniforms.metallicFactor = materialIn.metallicFactor;
                    renderObject->uniforms.roughnessFactor = materialIn.roughnessFactor;
                    renderObject->uniforms.emissiveFactor = materialIn.emissiveFactor;

                    renderObject->textures.baseColorTextureId = materialIn.baseColorTexture;
                    renderObject->textures.metallicRoughnessTextureId = materialIn.metallicRoughnessTexture;
                    renderObject->textures.normalTextureId = materialIn.normalTexture;

                    if (materialIn.baseColorTexture)
                    {
                        auto & textureInfo = common->textures[*materialIn.baseColorTexture];
                        textureInfo->users.push_back(renderObject);
                        textureInfo->sRGB = true;
                    }
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
                            glm::vec4(0.f, 0.f, 0.f, 1.f)
                        });

                        vertices.back().position = glm::vec3((nodeModelMatrix * glm::vec4(vertices.back().position, 1.f)));
                        vertices.back().normal = glm::normalize(glm::vec3((nodeModelMatrix * glm::vec4(vertices.back().normal, 0.f))));
                        vertices.back().tangent = glm::vec4(glm::normalize(glm::vec3((nodeModelMatrix * glm::vec4(glm::vec3(vertices.back().tangent), 0.f)))), vertices.back().tangent.w);

                        renderObject->bbox.expand(vertices.back().position);
                    }

                    auto indexIterator = glTF::AccessorIterator<std::uint16_t>(assetBufferData.data() + indexBufferView.byteOffset + indexAccessor.byteOffset, indexBufferView.byteStride);
                    for (int i = 0; i < indexAccessor.count; ++i)
                        indices.emplace_back(*indexIterator++);

                    if (materialIn.cloth)
                    {
                        // Make sure vertex count for cloth simulation is a multiple of workgroup size
                        int const vertexCountExtra = (32 - (renderObject->vertices.count % 32)) % 32;
                        int const vertexCount = renderObject->vertices.count + vertexCountExtra;

                        for (int i = 0; i < vertexCountExtra; ++i)
                            vertices.push_back({});

                        renderObject->cloth.emplace();

                        renderObject->cloth->edges.byteOffset = clothEdges.size() * sizeof(clothEdges[0]);
                        renderObject->cloth->edges.count = vertexCount * CLOTH_EDGES_PER_VERTEX;
                        renderObject->cloth->edges.byteLength = renderObject->cloth->edges.count * sizeof(clothEdges[0]);

                        renderObject->cloth->vertices.byteOffset = clothVertices.size() * sizeof(clothVertices[0]);
                        renderObject->cloth->vertices.count = vertexCount;
                        renderObject->cloth->vertices.byteLength = renderObject->cloth->vertices.count * sizeof(clothVertices[0]);

                        auto baseVertex = renderObject->vertices.byteOffset / sizeof(vertices[0]);
                        auto baseIndex = renderObject->indices.byteOffset / sizeof(indices[0]);

                        std::vector<std::vector<std::uint32_t>> edges(vertexCount);
                        for (int i = 0; i < indexAccessor.count; i += 3)
                        {
                            auto i0 = indices[baseIndex + i + 0];
                            auto i1 = indices[baseIndex + i + 1];
                            auto i2 = indices[baseIndex + i + 2];

                            edges[i0].push_back(i1);
                            edges[i0].push_back(i2);
                            edges[i1].push_back(i0);
                            edges[i1].push_back(i2);
                            edges[i2].push_back(i0);
                            edges[i2].push_back(i1);
                        }

                        std::vector<int> component(vertexCount, -1);
                        int componentCount = 0;
                        std::vector<int> componentVertices;

                        for (int i = 0; i < vertexCount; ++i)
                        {
                            if (component[i] != -1) continue;

                            component[i] = componentCount++;
                            componentVertices.push_back(1);

                            std::deque<int> queue;
                            queue.push_back(i);

                            while (!queue.empty())
                            {
                                auto v = queue.front();
                                queue.pop_front();

                                for (auto e : edges[v])
                                {
                                    if (component[e] == -1)
                                    {
                                        component[e] = component[v];
                                        queue.push_back(e);
                                        componentVertices.back()++;
                                    }
                                }
                            }
                        }

                        std::vector<int> componentTriangles(componentCount, 0);
                        std::vector<glm::vec3> avgComponentNormal(componentCount, glm::vec3(0.0));

                        for (int i = 0; i < indexAccessor.count; i += 3)
                        {
                            auto i0 = indices[baseIndex + i + 0];
                            auto i1 = indices[baseIndex + i + 1];
                            auto i2 = indices[baseIndex + i + 2];

                            auto c = component[i0];

                            componentTriangles[c] += 1;

                            auto p0 = vertices[baseVertex + i0].position;
                            auto p1 = vertices[baseVertex + i1].position;
                            auto p2 = vertices[baseVertex + i2].position;

                            auto n = glm::normalize(glm::cross(p1 - p0, p2 - p0));
                            avgComponentNormal[c] += n;
                        }

                        for (int c = 0; c < componentCount; ++c)
                            if (componentTriangles[c] > 0)
                                avgComponentNormal[c] /= (1.f * componentTriangles[c]);

                        // Sponza-specific heuristic: remove cloth mesh connected components which are
                        //    too small or face negative Z


                        std::vector<bool> frontFacing(vertexCount, false);

                        for (int i = 0; i < vertexCount; ++i)
                        {
                            int c = component[i];
                            frontFacing[i] = !(componentVertices[c] < 256 || avgComponentNormal[c].z < 0.f);
                        }

                        for (auto & vertexEdges : edges)
                        {
                            std::sort(vertexEdges.begin(), vertexEdges.end());
                            auto end = std::unique(vertexEdges.begin(), vertexEdges.end());
                            end = std::remove_if(vertexEdges.begin(), end, [&](auto i){ return !frontFacing[i]; });
                            vertexEdges.erase(end, vertexEdges.end());
                        }

                        std::vector<bool> disconnected(edges.size(), false);

                        for (int i = 0; i < edges.size(); ++i)
                        {
                            auto & vertexEdges = edges[i];

                            if (vertexEdges.size() == 2 && edges[vertexEdges[0]].size() == 2 && edges[vertexEdges[1]].size() == 2)
                                disconnected[i] = true;
                        }

                        float topY = -std::numeric_limits<float>::infinity();

                        for (int i = 0; i < edges.size(); ++i)
                        {
                            auto & vertexEdges = edges[i];

                            bool isTopVertex = true;
                            for (auto e : vertexEdges)
                            {
                                if (vertices[baseVertex + i].position.y < vertices[baseVertex + e].position.y)
                                {
                                    isTopVertex = false;
                                    break;
                                }
                            }

                            if (disconnected[i] || !frontFacing[i])
                            {
                                vertexEdges.assign(CLOTH_EDGES_PER_VERTEX, std::uint32_t(-1));
                                vertices[baseVertex + i].position = {0.f, 0.f, 0.f};
                            }
                            else if (isTopVertex)
                            {
                                vertexEdges.assign(CLOTH_EDGES_PER_VERTEX, std::uint32_t(-1));
                                topY = std::max(topY, vertices[baseVertex + i].position.y);
                            }
                            else if (vertexEdges.size() > CLOTH_EDGES_PER_VERTEX)
                            {
                                std::cout << "WARNING: " << vertexEdges.size() << " cloth edges is clamped to " << CLOTH_EDGES_PER_VERTEX;
                                vertexEdges.resize(CLOTH_EDGES_PER_VERTEX);
                            }
                            else while (vertexEdges.size() < CLOTH_EDGES_PER_VERTEX)
                            {
                                vertexEdges.push_back(-1);
                            }

                            for (auto e : vertexEdges)
                            {
                                ClothEdge edge;
                                edge.delta = glm::vec4(0.f);
                                edge.id = e;
                                if (e != std::uint32_t(-1))
                                {
                                    auto delta = vertices[baseVertex + e].position - vertices[baseVertex + i].position;
                                    edge.delta = glm::vec4(delta, glm::length(delta));
                                }
                                clothEdges.push_back(edge);
                            }
                        }

                        // Initial curtain shape, fit to the Sponza scene
                        for (int i = 0; i < renderObject->vertices.count; ++i)
                        {
                            auto & position = vertices[baseVertex + i].position;

                            if (!frontFacing[i])
                            {
                                position = glm::vec3(0.f);
                            }
                            else
                            {
//                                float distance = topY - position.y;
//                                float radius = 2.f;
//                                float angle = distance / radius;
//                                position.y = topY + radius * (1.f - std::cos(angle));
//                                position.z += radius * std::sin(angle)  * (position.z < 0.f ? 1.f : -1.f);
                            }
                        }

                        for (int i = 0; i < vertexCount; ++i)
                            clothVertices.push_back({
                                    .oldVelocity = glm::vec3(0.f),
                                    .velocity = glm::vec3(0.f),
                                    .newPosition = vertices[baseVertex + i].position,
                                });
                    }

                    renderObject->createTexturesBindGroup(device_, texturesBindGroupLayout_, defaultSampler_);

                    auto fixAlignment = [&](auto & buffer)
                    {
                        while (((buffer.size() * sizeof(buffer[0])) % minStorageBufferOffsetAlignment_) != 0)
                            buffer.push_back({});
                    };

                    fixAlignment(vertices);
                    fixAlignment(indices);
                    fixAlignment(clothVertices);
                    fixAlignment(clothEdges);
                }
            }
        }

        // Respect COPY_BUFFER_ALIGNMENT
        if ((indices.size() % 2) != 0)
            indices.push_back(0);

        WGPUBufferDescriptor vertexBufferDescriptor;
        vertexBufferDescriptor.nextInChain = nullptr;
        vertexBufferDescriptor.label = nullptr;
        vertexBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage;
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

        WGPUBufferDescriptor clothVerticesBufferDescriptor;
        clothVerticesBufferDescriptor.nextInChain = nullptr;
        clothVerticesBufferDescriptor.label = nullptr;
        clothVerticesBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage;
        clothVerticesBufferDescriptor.size = clothVertices.size() * sizeof(clothVertices[0]);
        clothVerticesBufferDescriptor.mappedAtCreation = false;

        common->clothVertexBuffer = wgpuDeviceCreateBuffer(device_, &clothVerticesBufferDescriptor);

        wgpuQueueWriteBuffer(queue_, common->clothVertexBuffer, 0, clothVertices.data(), clothVertices.size() * sizeof(clothVertices[0]));

        WGPUBufferDescriptor clothEdgesBufferDescriptor;
        clothEdgesBufferDescriptor.nextInChain = nullptr;
        clothEdgesBufferDescriptor.label = nullptr;
        clothEdgesBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage;
        clothEdgesBufferDescriptor.size = clothEdges.size() * sizeof(clothEdges[0]);
        clothEdgesBufferDescriptor.mappedAtCreation = false;

        common->clothEdgesBuffer = wgpuDeviceCreateBuffer(device_, &clothEdgesBufferDescriptor);

        wgpuQueueWriteBuffer(queue_, common->clothEdgesBuffer, 0, clothEdges.data(), clothEdges.size() * sizeof(clothEdges[0]));
    }

    for (auto & renderObject : result)
        if (renderObject->cloth)
            renderObject->createClothBindGroup(device_, simulateClothBindGroupLayout_, clothSettingsUniformBuffer_);

    for (std::uint32_t i = 0; i < common->textures.size(); ++i)
        loaderQueue_.push([this, common, i]{ loadTexture(*common->textures[i]); });

    return result;
}

void Engine::Impl::simulateCloth(std::vector<RenderObjectPtr> const & objects, Camera const & camera, Settings const & settings, int iterations)
{
    {
        ClothSettingsUniform settingsUniform;
        settingsUniform.dt = std::min(0.001f, settings.dt / iterations);
        settingsUniform.gravity = settings.gravity ? 10.f : 0.f;
        wgpuQueueWriteBuffer(queue_, clothSettingsUniformBuffer_, 0, &settingsUniform, sizeof(settingsUniform));
    }

    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPUComputePassEncoder computePass = createComputePass(commandEncoder);

    wgpuComputePassEncoderSetBindGroup(computePass, 1, cameraBindGroup_, 0, nullptr);

    for (auto const & object : objects)
    {
        if (!object->cloth) continue;

        wgpuComputePassEncoderSetBindGroup(computePass, 0, object->clothBindGroup, 0, nullptr);
        for (int i = 0; i < iterations; ++i)
        {
            wgpuComputePassEncoderSetPipeline(computePass, simulateClothPipeline_);
            wgpuComputePassEncoderDispatchWorkgroups(computePass, object->cloth->vertices.count / 32, 1, 1);

            wgpuComputePassEncoderSetPipeline(computePass, simulateClothCopyPipeline_);
            wgpuComputePassEncoderDispatchWorkgroups(computePass, object->cloth->vertices.count / 32, 1, 1);
        }

        if (iterations == 0)
        {
            wgpuComputePassEncoderSetPipeline(computePass, simulateClothCopyPipeline_);
            wgpuComputePassEncoderDispatchWorkgroups(computePass, object->cloth->vertices.count / 32, 1, 1);
        }
    }

    wgpuComputePassEncoderEnd(computePass);
    wgpuComputePassEncoderRelease(computePass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::renderShadow(std::vector<RenderObjectPtr> const & objects)
{
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createShadowRenderPass(commandEncoder, shadowMapView_, shadowMapDepthView_);

    wgpuRenderPassEncoderSetPipeline(renderPass, shadowPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, cameraBindGroup_, 0, nullptr);

    for (int i = 0; i < objects.size(); ++i)
    {
        auto const & object = objects[i];
        if (object->light) continue;

        std::uint32_t dynamicOffset = i * objectUniformBufferStride_;

        wgpuRenderPassEncoderSetBindGroup(renderPass, 1, objectBindGroup_, 1, &dynamicOffset);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 2, object->texturesBindGroup, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, object->common->vertexBuffer, object->vertices.byteOffset, object->vertices.byteLength);
        wgpuRenderPassEncoderSetIndexBuffer(renderPass, object->common->indexBuffer, object->indexFormat, object->indices.byteOffset, object->indices.byteLength);
        wgpuRenderPassEncoderDrawIndexed(renderPass, object->indices.count, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(renderPass);
    wgpuRenderPassEncoderRelease(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::blurShadow()
{
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPUComputePassEncoder computePass = createComputePass(commandEncoder);

    int const textureSize = wgpuTextureGetWidth(shadowMap_);
    int const workgroups = textureSize / 32;

    wgpuComputePassEncoderSetPipeline(computePass, blurShadowXPipeline_);
    wgpuComputePassEncoderSetBindGroup(computePass, 0, blurShadowXBindGroup_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(computePass, workgroups, textureSize, 1);

    wgpuComputePassEncoderSetPipeline(computePass, blurShadowYPipeline_);
    wgpuComputePassEncoderSetBindGroup(computePass, 0, blurShadowYBindGroup_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(computePass, textureSize, workgroups, 1);

    wgpuComputePassEncoderEnd(computePass);
    wgpuComputePassEncoderRelease(computePass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::renderEnv()
{
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createEnvRenderPass(commandEncoder, hdrMultisampleTextureView_, hdrResolveTextureView_);
    wgpuRenderPassEncoderSetPipeline(renderPass, envPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, cameraBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 1, emptyBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 2, emptyBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 3, lightsBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(renderPass);

    wgpuRenderPassEncoderRelease(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::renderMain(std::vector<RenderObjectPtr> const & objects)
{
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createMainRenderPass(commandEncoder, hdrMultisampleTextureView_, multisampleDepthTextureView_, hdrResolveTextureView_, glm::vec4(1.f));
    wgpuRenderPassEncoderSetPipeline(renderPass, mainPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, cameraBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 3, lightsBindGroup_, 0, nullptr);

    for (int i = 0; i < objects.size(); ++i)
    {
        auto const & object = objects[i];
        if (object->light) continue;

        std::uint32_t dynamicOffset = i * objectUniformBufferStride_;

        wgpuRenderPassEncoderSetBindGroup(renderPass, 1, objectBindGroup_, 1, &dynamicOffset);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 2, object->texturesBindGroup, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, object->common->vertexBuffer, object->vertices.byteOffset, object->vertices.byteLength);
        wgpuRenderPassEncoderSetIndexBuffer(renderPass, object->common->indexBuffer, object->indexFormat, object->indices.byteOffset, object->indices.byteLength);
        wgpuRenderPassEncoderDrawIndexed(renderPass, object->indices.count, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(renderPass);
    wgpuRenderPassEncoderRelease(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::renderWater(Settings const & settings)
{
    if (!waterGridVertexBuffer_)
        return;

    if (!waterBindGroup_)
        waterBindGroup_ = createWaterBindGroup(device_, waterBindGroupLayout_, hdrResolveTextureView_, multisampleDepthTextureView_, waterDataTextureView1_, waterUniformBuffer_);

    if (!simulateWaterBindGroup1_)
        simulateWaterBindGroup1_ = createSimulateWaterBindGroup(device_, simulateWaterBindGroupLayout_, waterUniformBuffer_, waterDataTextureView1_, waterDataTextureView2_);

    if (!simulateWaterBindGroup2_)
        simulateWaterBindGroup2_ = createSimulateWaterBindGroup(device_, simulateWaterBindGroupLayout_, waterUniformBuffer_, waterDataTextureView2_, waterDataTextureView1_);

    {
        WaterUniform waterUniform;
        waterUniform.cellSize = waterCellSize_;
        waterUniform.dt = std::min(0.016f, settings.dt);
        wgpuQueueWriteBuffer(queue_, waterUniformBuffer_, 0, &waterUniform, sizeof(waterUniform));
    }

    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    if (!settings.paused && settings.water)
    {
        WGPUComputePassEncoder computePass = createComputePass(commandEncoder);

        wgpuComputePassEncoderSetPipeline(computePass, simulateWaterPipeline_);

        for (int i = 0; i < 4; ++i)
        {
            wgpuComputePassEncoderSetBindGroup(computePass, 0, (i % 2) == 0 ? simulateWaterBindGroup1_ : simulateWaterBindGroup2_, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(computePass, waterCellCount_.x / 8, waterCellCount_.y / 8, 1);
        }

        wgpuComputePassEncoderEnd(computePass);
        wgpuComputePassEncoderRelease(computePass);
    }

    {
        WGPUImageCopyTexture source;
        source.nextInChain = nullptr;
        source.texture = hdrResolveTexture_;
        source.mipLevel = 0;
        source.origin = {0, 0, 0};
        source.aspect = WGPUTextureAspect_All;

        WGPUImageCopyTexture destination;
        destination.nextInChain = nullptr;
        destination.texture = hdrWaterTexture_;
        destination.mipLevel = 0;
        destination.origin = {0, 0, 0};
        destination.aspect = WGPUTextureAspect_All;

        WGPUExtent3D copySize;
        copySize.width = wgpuTextureGetWidth(hdrWaterTexture_);
        copySize.height = wgpuTextureGetHeight(hdrWaterTexture_);
        copySize.depthOrArrayLayers = 1;

        wgpuCommandEncoderCopyTextureToTexture(commandEncoder, &source, &destination, &copySize);
    }

    if (settings.water)
    {
        WGPURenderPassEncoder renderPass = createWaterRenderPass(commandEncoder, hdrWaterTextureView_, glm::vec4(0.f));
        wgpuRenderPassEncoderSetPipeline(renderPass, renderWaterPipeline_);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 0, cameraBindGroup_, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 1, emptyBindGroup_, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 2, emptyBindGroup_, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 3, lightsBindGroup_, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 4, waterBindGroup_, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, waterGridVertexBuffer_, 0, wgpuBufferGetSize(waterGridVertexBuffer_));
        wgpuRenderPassEncoderSetIndexBuffer(renderPass, waterGridIndexBuffer_, WGPUIndexFormat_Uint32, 0, wgpuBufferGetSize(waterGridIndexBuffer_));
        wgpuRenderPassEncoderDrawIndexed(renderPass, (waterCellCount_.x - 1) * (waterCellCount_.y - 1) * 6, 1, 0, 0, 0);

        wgpuRenderPassEncoderEnd(renderPass);
        wgpuRenderPassEncoderRelease(renderPass);
    }

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::renderLDR(WGPUTextureView targetView, WGPUTextureFormat surfaceFormat)
{
    if (!ldrPipeline_)
        ldrPipeline_ = createLDRPipeline(device_, ldrPipelineLayout_, ldrShaderModule_, surfaceFormat);

    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createLDRRenderPass(commandEncoder, targetView);
    wgpuRenderPassEncoderSetPipeline(renderPass, ldrPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, hdrBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(renderPass);
    wgpuRenderPassEncoderRelease(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::renderText(WGPUTextureView targetView, WGPUTextureFormat surfaceFormat, glm::ivec2 const & viewportSize, std::string const & text)
{
    if (!textPipeline_)
        textPipeline_ = createTextPipeline(device_, textPipelineLayout_, textShaderModule_, surfaceFormat);

    glm::ivec2 const fontSize{9, 12};
    glm::vec2 const fontSizef = glm::vec2(fontSize);
    glm::vec2 const cellSizef = fontSizef + glm::vec2(2.0, 2.0);
    glm::ivec2 const textureSize = (fontSize + glm::ivec2(2)) * glm::ivec2(16, 6);
    glm::vec2 const textureSizef = glm::vec2(textureSize);
    float const scale = 1.f;

    int vertexCount = 0;

    {
        std::vector<TextVertex> vertices;

        glm::vec2 penStart = fontSizef * 1.f * scale;
        glm::vec2 pen = penStart;

        for (unsigned char ch : text)
        {
            if (ch == '\n')
            {
                pen.x = penStart.x;
                pen.y += fontSizef.y * scale;
                continue;
            }

            if (ch < 32 || ch >= 128) ch = ' ';
            ch -= 32;

            TextVertex v00, v01, v10, v11;

            v00.position = pen + fontSizef * glm::vec2(0.0, 0.0) * scale;
            v01.position = pen + fontSizef * glm::vec2(1.0, 0.0) * scale;
            v10.position = pen + fontSizef * glm::vec2(0.0, 1.0) * scale;
            v11.position = pen + fontSizef * glm::vec2(1.0, 1.0) * scale;

            glm::vec2 const id{ch % 16, ch / 16};

            v00.texcoord = ((id + glm::vec2(0.0, 0.0)) * cellSizef + glm::vec2( 1.0,  1.0)) / textureSizef;
            v01.texcoord = ((id + glm::vec2(1.0, 0.0)) * cellSizef + glm::vec2(-1.0,  1.0)) / textureSizef;
            v10.texcoord = ((id + glm::vec2(0.0, 1.0)) * cellSizef + glm::vec2( 1.0, -1.0)) / textureSizef;
            v11.texcoord = ((id + glm::vec2(1.0, 1.0)) * cellSizef + glm::vec2(-1.0, -1.0)) / textureSizef;

            vertices.insert(vertices.end(), {v00, v01, v10, v10, v01, v11});

            pen.x += fontSizef.x * scale;
        }

        if (!textVertexBuffer_ || wgpuBufferGetSize(textVertexBuffer_) < vertices.size() * sizeof(vertices[0]))
        {
            WGPUBufferDescriptor descriptor;
            descriptor.nextInChain = nullptr;
            descriptor.label = nullptr;
            descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
            descriptor.size = vertices.size() * sizeof(vertices[0]);
            descriptor.mappedAtCreation = false;

            textVertexBuffer_ = wgpuDeviceCreateBuffer(device_, &descriptor);
        }

        wgpuQueueWriteBuffer(queue_, textVertexBuffer_, 0, vertices.data(), vertices.size() * sizeof(vertices[0]));

        vertexCount = vertices.size();
    }

    {
        TextUniform textUniform;
        textUniform.viewportSize = viewportSize;
        wgpuQueueWriteBuffer(queue_, textUniformBuffer_, 0, &textUniform, sizeof(textUniform));
    }

    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createTextRenderPass(commandEncoder, targetView);
    wgpuRenderPassEncoderSetPipeline(renderPass, textPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, textBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, textVertexBuffer_, 0, wgpuBufferGetSize(textVertexBuffer_));
    wgpuRenderPassEncoderDraw(renderPass, vertexCount, 1, 0, 0);

    wgpuRenderPassEncoderEnd(renderPass);
    wgpuRenderPassEncoderRelease(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
}

void Engine::Impl::updateFrameBuffer(glm::uvec2 const & renderTargetSize)
{
    if (!hdrMultisampleTexture_ || cachedRenderTargetSize_ != renderTargetSize)
    {
        if (hdrMultisampleTexture_)
        {
            wgpuTextureViewRelease(hdrMultisampleTextureView_);
            wgpuTextureRelease(hdrMultisampleTexture_);
            wgpuTextureViewRelease(multisampleDepthTextureView_);
            wgpuTextureRelease(multisampleDepthTexture_);
        }

        {
            WGPUTextureDescriptor descriptor;
            descriptor.nextInChain = nullptr;
            descriptor.label = nullptr;
            descriptor.usage = WGPUTextureUsage_RenderAttachment;
            descriptor.dimension = WGPUTextureDimension_2D;
            descriptor.size = {renderTargetSize.x, renderTargetSize.y, 1};
            descriptor.format = WGPUTextureFormat_RGBA16Float;
            descriptor.mipLevelCount = 1;
            descriptor.sampleCount = 4;
            descriptor.viewFormatCount = 0;
            descriptor.viewFormats = nullptr;

            hdrMultisampleTexture_ = wgpuDeviceCreateTexture(device_, &descriptor);
            hdrMultisampleTextureView_ = createTextureView(hdrMultisampleTexture_);
        }

        {
            WGPUTextureDescriptor descriptor;
            descriptor.nextInChain = nullptr;
            descriptor.label = nullptr;
            descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc;
            descriptor.dimension = WGPUTextureDimension_2D;
            descriptor.size = {renderTargetSize.x, renderTargetSize.y, 1};
            descriptor.format = WGPUTextureFormat_RGBA16Float;
            descriptor.mipLevelCount = 1;
            descriptor.sampleCount = 1;
            descriptor.viewFormatCount = 0;
            descriptor.viewFormats = nullptr;

            hdrResolveTexture_ = wgpuDeviceCreateTexture(device_, &descriptor);
            hdrResolveTextureView_ = createTextureView(hdrResolveTexture_);
        }

        {
            WGPUTextureDescriptor descriptor;
            descriptor.nextInChain = nullptr;
            descriptor.label = nullptr;
            descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
            descriptor.dimension = WGPUTextureDimension_2D;
            descriptor.size = {renderTargetSize.x, renderTargetSize.y, 1};
            descriptor.format = WGPUTextureFormat_RGBA16Float;
            descriptor.mipLevelCount = 1;
            descriptor.sampleCount = 1;
            descriptor.viewFormatCount = 0;
            descriptor.viewFormats = nullptr;

            hdrWaterTexture_ = wgpuDeviceCreateTexture(device_, &descriptor);
            hdrWaterTextureView_ = createTextureView(hdrWaterTexture_);
        }

        {
            WGPUTextureDescriptor descriptor;
            descriptor.nextInChain = nullptr;
            descriptor.label = nullptr;
            descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
            descriptor.dimension = WGPUTextureDimension_2D;
            descriptor.size = {renderTargetSize.x, renderTargetSize.y, 1};
            descriptor.format = WGPUTextureFormat_Depth24Plus;
            descriptor.mipLevelCount = 1;
            descriptor.sampleCount = 4;
            descriptor.viewFormatCount = 0;
            descriptor.viewFormats = nullptr;

            multisampleDepthTexture_ = wgpuDeviceCreateTexture(device_, &descriptor);
            multisampleDepthTextureView_ = createTextureView(multisampleDepthTexture_);
        }

        if (hdrBindGroup_)
            wgpuBindGroupRelease(hdrBindGroup_);

        hdrBindGroup_ = createHDRBindGroup(device_, hdrBindGroupLayout_, hdrWaterTextureView_);

        if (waterBindGroup_)
            wgpuBindGroupRelease(waterBindGroup_);
        waterBindGroup_ = nullptr;

        cachedRenderTargetSize_ = renderTargetSize;
    }
}

void Engine::Impl::updateCameraBuffer(Camera const & camera, Settings const & settings, WGPUBuffer buffer, glm::ivec2 const & viewport)
{
    CameraUniform cameraUniform;
    cameraUniform.viewProjection = glToVkProjection(camera.viewProjectionMatrix());
    cameraUniform.viewProjectionInverse = glm::inverse(cameraUniform.viewProjection);
    cameraUniform.position = camera.position();
    cameraUniform.shock = glm::vec4(settings.shockCenter, settings.shockDistance);
    cameraUniform.shockDirection = settings.shockDirection;
    cameraUniform.time = settings.time;

    wgpuQueueWriteBuffer(queue_, buffer, 0, &cameraUniform, sizeof(CameraUniform));
}

void Engine::Impl::updateObjectUniformBuffer(std::vector<RenderObjectPtr> const & objects)
{
    if (!objectUniformBuffer_ || wgpuBufferGetSize(objectUniformBuffer_) < objects.size() * objectUniformBufferStride_)
    {
        if (objectUniformBuffer_)
        {
            wgpuBindGroupRelease(objectBindGroup_);
            wgpuBufferRelease(objectUniformBuffer_);
        }

        objectUniformBuffer_ = createUniformBuffer(device_, objects.size() * objectUniformBufferStride_);

        objectBindGroup_ = createObjectBindGroup(device_, objectBindGroupLayout_, objectUniformBuffer_);

        for (auto const & object : objects)
            if (!object->light)
                object->createTexturesBindGroup(device_, texturesBindGroupLayout_, defaultSampler_);
    }

    std::vector<char> objectUniforms(objects.size() * objectUniformBufferStride_);
    for (int i = 0; i < objects.size(); ++i)
        std::memcpy(objectUniforms.data() + i * objectUniformBufferStride_, &objects[i]->uniforms, sizeof(objects[i]->uniforms));

    wgpuQueueWriteBuffer(queue_, objectUniformBuffer_, 0, objectUniforms.data(), objectUniforms.size() * sizeof(objectUniforms[0]));
}

int Engine::Impl::updatePointLightsBuffer(std::vector<RenderObjectPtr> const & objects)
{
    std::vector<PointLight> lights;
    for (auto const & object : objects)
    {
        if (object->light)
        {
            auto & light = lights.emplace_back();
            light.position = object->light->position;
            light.intensity = object->light->intensity;
        }
    }

    if (lights.size() * sizeof(lights[0]) > wgpuBufferGetSize(pointLightsBuffer_))
    {
        wgpuBufferRelease(pointLightsBuffer_);
        pointLightsBuffer_ = createStorageBuffer(device_, lights.size() * sizeof(lights[0]));
        recreateLightsBindGroup();
    }

    wgpuQueueWriteBuffer(queue_, pointLightsBuffer_, 0, lights.data(), lights.size() * sizeof(lights[0]));

    return lights.size();
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

void Engine::Impl::updateLightsUniformBuffer(glm::mat4 const & shadowProjection, Settings const & settings, int pointLightCount)
{
    LightsUniform lightsUniform;
    lightsUniform.shadowProjection = shadowProjection;
    lightsUniform.ambientLight = settings.ambientLight;
    lightsUniform.envIntensity = settings.envIntensity;
    lightsUniform.sunDirection = settings.sunDirection;
    lightsUniform.sunIntensity = settings.sunIntensity;
    lightsUniform.pointLightCount = pointLightCount;

    wgpuQueueWriteBuffer(queue_, lightsUniformBuffer_, 0, &lightsUniform, sizeof(LightsUniform));
}

void Engine::Impl::loadTexture(RenderObjectCommon::TextureInfo & textureInfo)
{
    auto imageInfo = glTF::loadImage(textureInfo.assetPath, textureInfo.uri);

    WGPUTextureDescriptor textureDescriptor;
    textureDescriptor.nextInChain = nullptr;
    textureDescriptor.label = nullptr;
    textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding;
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

    {
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

        wgpuQueueWriteTexture(queue_, &imageCopyTexture, imageInfo.data.get(), imageInfo.width * imageInfo.height * imageInfo.channels, &textureDataLayout, &writeSize);
    }

    std::vector<WGPUBindGroup> bindGroups;

    bool const useSRGB = (textureInfo.sRGB && sRGBViewFormat);
    WGPUComputePipeline pipeline = useSRGB ? genMipmapSRGBPipeline_ : genMipmapPipeline_;

    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);
    WGPUComputePassEncoder computePass = createComputePass(commandEncoder);
    wgpuComputePassEncoderSetPipeline(computePass, pipeline);

    for (int level = 1; level < textureDescriptor.mipLevelCount; ++level)
    {
        auto inputView = createTextureView(texture, level - 1);
        auto outputView = createTextureView(texture, level);
        auto bindGroup = createGenMipmapBindGroup(device_, genMipmapBindGroupLayout_, inputView, outputView);

        int workgroupCountX = std::ceil((imageInfo.width >> level) / 8.f);
        int workgroupCountY = std::ceil((imageInfo.height >> level) / 8.f);

        wgpuComputePassEncoderSetBindGroup(computePass, 0, bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(computePass, workgroupCountX, workgroupCountY, 1);

        bindGroups.push_back(bindGroup);
        wgpuTextureViewRelease(outputView);
        wgpuTextureViewRelease(inputView);
    }

    wgpuComputePassEncoderEnd(computePass);
    wgpuComputePassEncoderRelease(computePass);

    // Bind groups should live as long as the compute pass lives,
    // so we defer it's release until the compute pass ends
    for (auto bindGroup : bindGroups)
        wgpuBindGroupRelease(bindGroup);

    WGPUCommandBuffer commandBuffer = commandEncoderFinish(commandEncoder);

    // This is probably a bug in wgpu, but submitting a compute pass
    // in a separate thread causes GPU hangs or device loss, see
    // https://github.com/gfx-rs/wgpu/issues/4877
    renderQueue_.push([this, commandBuffer]
    {
        wgpuQueueSubmit(queue_, 1, &commandBuffer);
    });

    textureInfo.texture.store(texture);

    for (auto const & user : textureInfo.users)
        renderQueue_.push([this, user]{
            if (auto object = user.lock())
                object->createTexturesBindGroup(device_, texturesBindGroupLayout_, defaultSampler_);
        });
}

void Engine::Impl::recreateLightsBindGroup()
{
    if (lightsBindGroup_)
        wgpuBindGroupRelease(lightsBindGroup_);

    lightsBindGroup_ = createLightsBindGroup(device_, lightsBindGroupLayout_, lightsUniformBuffer_,
        shadowSampler_, shadowMapView_, envSampler_, envTextureView_, pointLightsBuffer_, noise3DTextureView_, noise3DSampler_);
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

Engine::Engine(WGPUDevice device, WGPUQueue queue, std::filesystem::path const & noise3DPath, std::filesystem::path const & fontPath)
    : pimpl_(std::make_unique<Impl>(device, queue, noise3DPath, fontPath))
{}

Engine::~Engine() = default;

std::vector<RenderObjectPtr> Engine::loadGLTF(std::filesystem::path const & assetPath)
{
    return pimpl_->loadGLTF(assetPath);
}

void Engine::setEnvMap(std::filesystem::path const & hdrImagePath)
{
    pimpl_->setEnvMap(hdrImagePath);
}

void Engine::setWater(Box const & box, std::filesystem::path const & waterState)
{
    pimpl_->setWater(box, waterState);
}

void Engine::render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, Settings const & settings, std::string const & text)
{
    pimpl_->render(target, objects, camera, sceneBbox, settings, text);
}

Box Engine::bbox(std::vector<RenderObjectPtr> const & objects) const
{
    Box result;

    for (auto const & object : objects)
        result.expand(object->bbox);

    return result;
}

