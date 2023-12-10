#include <webgpu.h>

#include <webgpu-demo/sdl_wgpu.h>
#include <webgpu-demo/application.hpp>
#include <webgpu-demo/camera.hpp>
#include <webgpu-demo/gltf_loader.hpp>

#include <glm/glm.hpp>

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

static const char shaderCode[] =
R"(

@group(0) @binding(0) var<uniform> viewProjection: mat4x4f;

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

@vertex
fn vertexMain(in : VertexInput) -> VertexOutput {
    return VertexOutput(viewProjection * vec4f(in.position, 1.0), in.normal, in.tangent, in.texcoord);
}

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4f {
    return vec4f(pow(in.normal * 0.5 + vec3f(0.5), vec3f(2.2)), 1.0);
}

)";

struct RenderObject
{
    std::uint32_t vertexByteOffset;
    std::uint32_t vertexByteLength;
    std::uint32_t vertexCount;

    std::uint32_t indexByteOffset;
    std::uint32_t indexByteLength;
    std::uint32_t indexCount;

    WGPUIndexFormat indexFormat;
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

    WGPUBindGroupLayoutEntry bindGroupLayoutEntry[1];
    bindGroupLayoutEntry[0].nextInChain = nullptr;
    bindGroupLayoutEntry[0].binding = 0;
    bindGroupLayoutEntry[0].visibility = WGPUShaderStage_Vertex;
    bindGroupLayoutEntry[0].buffer.nextInChain = nullptr;
    bindGroupLayoutEntry[0].buffer.type = WGPUBufferBindingType_Uniform;
    bindGroupLayoutEntry[0].buffer.hasDynamicOffset = false;
    bindGroupLayoutEntry[0].buffer.minBindingSize = 64;
    bindGroupLayoutEntry[0].sampler.nextInChain = nullptr;
    bindGroupLayoutEntry[0].sampler.type = WGPUSamplerBindingType_Undefined;
    bindGroupLayoutEntry[0].texture.nextInChain = nullptr;
    bindGroupLayoutEntry[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    bindGroupLayoutEntry[0].texture.multisampled = false;
    bindGroupLayoutEntry[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    bindGroupLayoutEntry[0].storageTexture.nextInChain = nullptr;
    bindGroupLayoutEntry[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    bindGroupLayoutEntry[0].storageTexture.format = WGPUTextureFormat_Undefined;
    bindGroupLayoutEntry[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor bindGroupLayoutDescriptor;
    bindGroupLayoutDescriptor.nextInChain = nullptr;
    bindGroupLayoutDescriptor.label = nullptr;
    bindGroupLayoutDescriptor.entryCount = 1;
    bindGroupLayoutDescriptor.entries = bindGroupLayoutEntry;

    WGPUBindGroupLayout bindGroupLayout = wgpuDeviceCreateBindGroupLayout(application.device(), &bindGroupLayoutDescriptor);

    WGPUPipelineLayoutDescriptor pipelineLayoutDescriptor;
    pipelineLayoutDescriptor.nextInChain = nullptr;
    pipelineLayoutDescriptor.label = nullptr;
    pipelineLayoutDescriptor.bindGroupLayoutCount = 1;
    pipelineLayoutDescriptor.bindGroupLayouts = &bindGroupLayout;

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
    renderPipelineDescriptor.primitive.cullMode = WGPUCullMode_Back;
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

    {
        if (asset.buffers.size() != 1)
            throw std::runtime_error("Only 1 binary buffer is supported");

        std::vector<char> assetBufferData = glTF::loadBuffer(assetPath, asset.buffers[0].uri);

        std::vector<Vertex> vertices;
        std::vector<std::uint16_t> indices;

        for (auto const & mesh : asset.meshes)
        {
            for (auto const & primitive : mesh.primitives)
            {
                if (!primitive.attributes.position) continue;
                if (!primitive.attributes.normal) continue;
                if (!primitive.attributes.tangent) continue;
                if (!primitive.attributes.texcoord) continue;
                if (!primitive.indices) continue;

                if (primitive.mode != glTF::Primitive::Mode::Triangles) continue;

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

                auto & renderObject = renderObjects.emplace_back();

                renderObject.vertexByteOffset = vertices.size() * sizeof(vertices[0]);
                renderObject.vertexByteLength = positionAccessor.count * sizeof(vertices[0]);
                renderObject.vertexCount = positionAccessor.count;
                renderObject.indexByteOffset = indices.size() * sizeof(indices[0]);
                renderObject.indexByteLength = indexAccessor.count * sizeof(indices[0]);
                renderObject.indexCount = indexAccessor.count;
                renderObject.indexFormat = WGPUIndexFormat_Uint16;

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
                    vertices.push_back({
                        *positionIterator++,
                        *  normalIterator++,
                        * tangentIterator++,
                        *texcoordIterator++,
                    });

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

    WGPUBufferDescriptor uniformBufferDescriptor;
    uniformBufferDescriptor.nextInChain = nullptr;
    uniformBufferDescriptor.label = nullptr;
    uniformBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    uniformBufferDescriptor.size = 64;
    uniformBufferDescriptor.mappedAtCreation = false;

    WGPUBuffer uniformBuffer = wgpuDeviceCreateBuffer(application.device(), &uniformBufferDescriptor);

    WGPUBindGroupEntry bindGroupEntry;
    bindGroupEntry.nextInChain = nullptr;
    bindGroupEntry.binding = 0;
    bindGroupEntry.buffer = uniformBuffer;
    bindGroupEntry.offset = 0;
    bindGroupEntry.size = 64;
    bindGroupEntry.sampler = nullptr;
    bindGroupEntry.textureView = nullptr;

    WGPUBindGroupDescriptor bindGroupDescriptor;
    bindGroupDescriptor.nextInChain = nullptr;
    bindGroupDescriptor.label = nullptr;
    bindGroupDescriptor.layout = bindGroupLayout;
    bindGroupDescriptor.entryCount = 1;
    bindGroupDescriptor.entries = &bindGroupEntry;

    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(application.device(), &bindGroupDescriptor);

    WGPUTexture depthTexture = nullptr;
    WGPUTextureView depthTextureView = nullptr;

    Camera camera;
    camera.setFov(glm::radians(45.f), application.width() * 1.f / application.height());

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
        });

        glm::mat4 const viewProjectionMatrix = camera.viewProjectionMatrix();

        wgpuQueueWriteBuffer(application.queue(), uniformBuffer, 0, &viewProjectionMatrix, 64);

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
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroup, 0, nullptr);

        for (auto const & renderObject : renderObjects)
        {
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
