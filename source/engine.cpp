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

struct Engine::Impl
{
    Impl(WGPUDevice device, WGPUQueue queue);
    ~Impl();

    void setEnvMap(std::filesystem::path const & hdrImagePath);
    void render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, LightSettings const & lightSettings);
    std::vector<RenderObjectPtr> loadGLTF(std::filesystem::path const & assetPath);

private:
    WGPUDevice device_;
    WGPUQueue queue_;

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

    WGPUShaderModule shaderModule_;
    WGPUShaderModule genMipmapShaderModule_;

    WGPUTexture shadowMap_;
    WGPUTextureView shadowMapView_;

    WGPUSampler defaultSampler_;
    WGPUSampler shadowSampler_;

    WGPUPipelineLayout mainPipelineLayout_;
    WGPURenderPipeline mainPipeline_;
    WGPUPipelineLayout shadowPipelineLayout_;
    WGPURenderPipeline shadowPipeline_;
    WGPUPipelineLayout envPipelineLayout_;
    WGPURenderPipeline envPipeline_;
    WGPUPipelineLayout genMipmapPipelineLayout_;
    WGPUComputePipeline genMipmapPipeline_;
    WGPUComputePipeline genMipmapSRGBPipeline_;

    WGPUBuffer cameraUniformBuffer_;
    WGPUBuffer objectUniformBuffer_;
    WGPUBuffer lightsUniformBuffer_;

    WGPUTexture stubEnvTexture_;
    WGPUTexture envTexture_;
    WGPUTextureView envTextureView_;

    std::uint64_t objectUniformBufferStride_ = 256;

    WGPUBindGroup emptyBindGroup_;
    WGPUBindGroup cameraBindGroup_;
    WGPUBindGroup objectBindGroup_;
    WGPUBindGroup lightsBindGroup_;

    WGPUTexture frameTexture_;
    WGPUTextureView frameTextureView_;
    WGPUTexture depthTexture_;
    WGPUTextureView depthTextureView_;

    WGPUTexture whiteTexture_;

    glm::uvec2 cachedRenderTargetSize_{0, 0};

    void renderShadow(std::vector<RenderObjectPtr> const & objects);
    void renderEnv(WGPUTextureView targetView);
    void renderMain(std::vector<RenderObjectPtr> const & objects, WGPUTextureView targetView);

    void updateFrameBuffer(glm::uvec2 const & renderTargetSize, WGPUTextureFormat surfaceFormat);
    void updateCameraUniformBuffer(Camera const & camera);
    void updateObjectUniformBuffer(std::vector<RenderObjectPtr> const & objects);
    glm::mat4 computeShadowProjection(glm::vec3 const & lightDirection, Box const & sceneBbox);
    void updateCameraUniformBufferShadow(glm::mat4 const & shadowProjection);
    void updateLightsUniformBuffer(glm::mat4 const & shadowProjection, LightSettings const & lightSettings);
    void loadTexture(RenderObjectCommon::TextureInfo & textureInfo);
    void loaderThreadMain();
};

Engine::Impl::Impl(WGPUDevice device, WGPUQueue queue)
    : device_(device)
    , queue_(queue)
    , loaderThread_([this]{ loaderThreadMain(); })
    , emptyBindGroupLayout_(createEmptyBindGroupLayout(device_))
    , cameraBindGroupLayout_(createCameraBindGroupLayout(device_))
    , objectBindGroupLayout_(createObjectBindGroupLayout(device_))
    , texturesBindGroupLayout_(createTexturesBindGroupLayout(device_))
    , lightsBindGroupLayout_(createLightsBindGroupLayout(device_))
    , genMipmapBindGroupLayout_(createGenMipmapBindGroupLayout(device_))
    , shaderModule_(createShaderModule(device_, mainShader))
    , genMipmapShaderModule_(createShaderModule(device_, genMipmapShader))
    , shadowMap_(createShadowMapTexture(device_, 4096))
    , shadowMapView_(createTextureView(shadowMap_))
    , defaultSampler_(createDefaultSampler(device_))
    , shadowSampler_(createShadowSampler(device_))
    , mainPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, objectBindGroupLayout_, texturesBindGroupLayout_, lightsBindGroupLayout_}))
    , mainPipeline_(nullptr)
    , shadowPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, objectBindGroupLayout_, texturesBindGroupLayout_}))
    , shadowPipeline_(createShadowPipeline(device_, shadowPipelineLayout_, shaderModule_))
    , envPipelineLayout_(createPipelineLayout(device_, {cameraBindGroupLayout_, emptyBindGroupLayout_, emptyBindGroupLayout_, lightsBindGroupLayout_}))
    , envPipeline_(nullptr)
    , genMipmapPipelineLayout_(createPipelineLayout(device_, {genMipmapBindGroupLayout_}))
    , genMipmapPipeline_(createMipmapPipeline(device_, genMipmapPipelineLayout_, genMipmapShaderModule_))
    , genMipmapSRGBPipeline_(createMipmapSRGBPipeline(device_, genMipmapPipelineLayout_, genMipmapShaderModule_))
    , cameraUniformBuffer_(createUniformBuffer(device_, sizeof(CameraUniform)))
    , objectUniformBuffer_(nullptr)
    , lightsUniformBuffer_(createUniformBuffer(device_, sizeof(LightsUniform)))
    , stubEnvTexture_(createStubEnvTexture(device_, queue_))
    , envTexture_(nullptr)
    , envTextureView_(createTextureView(stubEnvTexture_))
    , emptyBindGroup_(createEmptyBindGroup(device_, emptyBindGroupLayout_))
    , cameraBindGroup_(createCameraBindGroup(device_, cameraBindGroupLayout_, cameraUniformBuffer_))
    , objectBindGroup_(nullptr)
    , lightsBindGroup_(createLightsBindGroup(device_, lightsBindGroupLayout_, lightsUniformBuffer_, shadowSampler_, shadowMapView_, defaultSampler_, envTextureView_))
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

void Engine::Impl::setEnvMap(std::filesystem::path const & hdrImagePath)
{
    loaderQueue_.push([this, hdrImagePath]{
        int width, height, channels;
        float * pixels = stbi_loadf(hdrImagePath.c_str(), &width, &height, &channels, 4);

        WGPUTextureDescriptor textureDescriptor;
        textureDescriptor.nextInChain = nullptr;
        textureDescriptor.label = nullptr;
        textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
        textureDescriptor.dimension = WGPUTextureDimension_2D;
        textureDescriptor.size = {(std::uint32_t)width, (std::uint32_t)height, 1};
        textureDescriptor.format = WGPUTextureFormat_RGBA32Float;
        textureDescriptor.mipLevelCount = 1;
        textureDescriptor.sampleCount = 1;
        textureDescriptor.viewFormatCount = 0;
        textureDescriptor.viewFormats = nullptr;

        WGPUTexture texture = wgpuDeviceCreateTexture(device_, &textureDescriptor);

        renderQueue_.push([this, width, height, pixels, texture]{
            if (envTexture_)
                wgpuTextureRelease(envTexture_);

            envTexture_ = texture;

            WGPUImageCopyTexture imageCopyTexture;
            imageCopyTexture.nextInChain = nullptr;
            imageCopyTexture.texture = envTexture_;
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

            envTextureView_ = createTextureView(envTexture_);

            if (lightsBindGroup_)
                wgpuBindGroupRelease(lightsBindGroup_);

            lightsBindGroup_ = createLightsBindGroup(device_, lightsBindGroupLayout_, lightsUniformBuffer_, shadowSampler_, shadowMapView_, defaultSampler_, envTextureView_);
        });
    });
}

void Engine::Impl::render(WGPUTexture target, std::vector<RenderObjectPtr> const & objects, Camera const & camera, Box const & sceneBbox, LightSettings const & lightSettings)
{
    for (auto task : renderQueue_.grab())
        task();

    WGPUTextureFormat surfaceFormat = wgpuTextureGetFormat(target);

    if (!mainPipeline_)
        mainPipeline_ = createMainPipeline(device_, mainPipelineLayout_, surfaceFormat, shaderModule_);

    if (!envPipeline_)
        envPipeline_ = createEnvPipeline(device_, envPipelineLayout_, surfaceFormat, shaderModule_);

    updateFrameBuffer({wgpuTextureGetWidth(target), wgpuTextureGetHeight(target)}, surfaceFormat);

    updateObjectUniformBuffer(objects);

    WGPUTextureView targetView = createTextureView(target);

    glm::mat4 shadowProjection = computeShadowProjection(lightSettings.sunDirection, sceneBbox);

    updateCameraUniformBufferShadow(shadowProjection);

    renderShadow(objects);

    updateCameraUniformBuffer(camera);
    updateLightsUniformBuffer(shadowProjection, lightSettings);

    renderEnv(targetView);
    renderMain(objects, targetView);

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

void Engine::Impl::renderShadow(std::vector<RenderObjectPtr> const & objects)
{
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createShadowRenderPass(commandEncoder, shadowMapView_);

    wgpuRenderPassEncoderSetPipeline(renderPass, shadowPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, cameraBindGroup_, 0, nullptr);

    for (int i = 0; i < objects.size(); ++i)
    {
        auto const & object = objects[i];

        std::uint32_t dynamicOffset = i * objectUniformBufferStride_;

        wgpuRenderPassEncoderSetBindGroup(renderPass, 1, objectBindGroup_, 1, &dynamicOffset);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 2, object->texturesBindGroup, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, object->common->vertexBuffer, object->vertexByteOffset, object->vertexByteLength);
        wgpuRenderPassEncoderSetIndexBuffer(renderPass, object->common->indexBuffer, object->indexFormat, object->indexByteOffset, object->indexByteLength);
        wgpuRenderPassEncoderDrawIndexed(renderPass, object->indexCount, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);
}

void Engine::Impl::renderEnv(WGPUTextureView targetView)
{
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createEnvRenderPass(commandEncoder, frameTextureView_, targetView);
    wgpuRenderPassEncoderSetPipeline(renderPass, envPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, cameraBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 1, emptyBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 2, emptyBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 3, lightsBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);
}

void Engine::Impl::renderMain(std::vector<RenderObjectPtr> const & objects, WGPUTextureView targetView)
{
    WGPUCommandEncoder commandEncoder = createCommandEncoder(device_);

    WGPURenderPassEncoder renderPass = createMainRenderPass(commandEncoder, frameTextureView_, depthTextureView_, targetView, glm::vec4(1.f));
    wgpuRenderPassEncoderSetPipeline(renderPass, mainPipeline_);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, cameraBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 3, lightsBindGroup_, 0, nullptr);

    for (int i = 0; i < objects.size(); ++i)
    {
        auto const & object = objects[i];

        std::uint32_t dynamicOffset = i * objectUniformBufferStride_;

        wgpuRenderPassEncoderSetBindGroup(renderPass, 1, objectBindGroup_, 1, &dynamicOffset);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 2, object->texturesBindGroup, 0, nullptr);

        wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, object->common->vertexBuffer, object->vertexByteOffset, object->vertexByteLength);
        wgpuRenderPassEncoderSetIndexBuffer(renderPass, object->common->indexBuffer, object->indexFormat, object->indexByteOffset, object->indexByteLength);
        wgpuRenderPassEncoderDrawIndexed(renderPass, object->indexCount, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(renderPass);

    auto commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);
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
            object->createTexturesBindGroup(device_, texturesBindGroupLayout_, defaultSampler_);
    }

    std::vector<char> objectUniforms(objects.size() * objectUniformBufferStride_);
    for (int i = 0; i < objects.size(); ++i)
        std::memcpy(objectUniforms.data() + i * objectUniformBufferStride_, &objects[i]->uniforms, sizeof(objects[i]->uniforms));

    wgpuQueueWriteBuffer(queue_, objectUniformBuffer_, 0, objectUniforms.data(), objectUniforms.size() * sizeof(objectUniforms[0]));
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

void Engine::Impl::updateLightsUniformBuffer(glm::mat4 const & shadowProjection, LightSettings const & lightSettings)
{
    LightsUniform lightsUniform;
    lightsUniform.shadowProjection = shadowProjection;
    lightsUniform.ambientLight = lightSettings.ambientLight;
    lightsUniform.envIntensity = lightSettings.envIntensity;
    lightsUniform.sunDirection = lightSettings.sunDirection;
    lightsUniform.sunIntensity = lightSettings.sunIntensity;

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

    // Bind groups should live as long as the compute pass lives,
    // so we defer it's release until the compute pass ends
    for (auto bindGroup : bindGroups)
        wgpuBindGroupRelease(bindGroup);

    WGPUCommandBuffer commandBuffer = commandEncoderFinish(commandEncoder);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

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

void Engine::setEnvMap(std::filesystem::path const & hdrImagePath)
{
    pimpl_->setEnvMap(hdrImagePath);
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

