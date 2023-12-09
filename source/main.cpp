#include <webgpu.h>

#include <webgpu-demo/sdl_wgpu.h>
#include <webgpu-demo/application.hpp>

#include <iostream>
#include <vector>
#include <cstdint>

struct Vertex
{
    float x, y;
    std::uint32_t color;
};

static const char shaderCode[] =
R"(

@group(0) @binding(0) var<uniform> viewProjection: mat4x4f;

struct VertexInput {
    @builtin(vertex_index) index : u32,
    @location(0) position : vec2f,
    @location(1) color : vec4f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) color : vec4f,
}

@vertex
fn vertexMain(in : VertexInput) -> VertexOutput {
    return VertexOutput(viewProjection * vec4f(in.position, 0.0, 1.0), in.color);
}

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4f {
    return pow(in.color, vec4f(1.0 / 2.2));
}

)";

int main()
{
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

    WGPUVertexAttribute attributes[2];
    attributes[0].format = WGPUVertexFormat_Float32x2;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Unorm8x4;
    attributes[1].offset = 8;
    attributes[1].shaderLocation = 1;

    WGPUVertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.arrayStride = sizeof(Vertex);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexBufferLayout.attributeCount = 2;
    vertexBufferLayout.attributes = attributes;

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
    renderPipelineDescriptor.depthStencil = nullptr;
    renderPipelineDescriptor.multisample.nextInChain = nullptr;
    renderPipelineDescriptor.multisample.count = 1;
    renderPipelineDescriptor.multisample.mask = -1;
    renderPipelineDescriptor.multisample.alphaToCoverageEnabled = false;
    renderPipelineDescriptor.fragment = &fragmentState;

    WGPURenderPipeline renderPipeline = wgpuDeviceCreateRenderPipeline(application.device(), &renderPipelineDescriptor);

    std::vector<Vertex> vertices
    {
        {-0.5f, -0.5f, 0xff0000ffu},
        { 0.5f, -0.5f, 0xff00ff00u},
        { 0.0f,  0.5f, 0xffff0000u},
    };

    WGPUBufferDescriptor vertexBufferDescriptor;
    vertexBufferDescriptor.nextInChain = nullptr;
    vertexBufferDescriptor.label = nullptr;
    vertexBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
    vertexBufferDescriptor.size = vertices.size() * sizeof(vertices[0]);
    vertexBufferDescriptor.mappedAtCreation = false;

    WGPUBuffer vertexBuffer = wgpuDeviceCreateBuffer(application.device(), &vertexBufferDescriptor);

    wgpuQueueWriteBuffer(application.queue(), vertexBuffer, 0, vertices.data(), vertices.size() * sizeof(vertices[0]));

    float viewProjectionMatrix[16] =
    {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f,
    };

    WGPUBufferDescriptor uniformBufferDescriptor;
    uniformBufferDescriptor.nextInChain = nullptr;
    uniformBufferDescriptor.label = nullptr;
    uniformBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    uniformBufferDescriptor.size = 64;
    uniformBufferDescriptor.mappedAtCreation = false;

    WGPUBuffer uniformBuffer = wgpuDeviceCreateBuffer(application.device(), &uniformBufferDescriptor);

    wgpuQueueWriteBuffer(application.queue(), uniformBuffer, 0, viewProjectionMatrix, 64);

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

    int frameId = 0;

    for (bool running = true; running;)
    {
        std::cout << "Frame " << frameId << std::endl;

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
                break;
            }
            break;
        }

        auto surfaceTextureView = application.nextSwapchainView();
        if (!surfaceTextureView)
        {
            ++frameId;
            continue;
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

        WGPURenderPassDescriptor renderPassDescriptor;
        renderPassDescriptor.nextInChain = nullptr;
        renderPassDescriptor.label = nullptr;
        renderPassDescriptor.colorAttachmentCount = 1;
        renderPassDescriptor.colorAttachments = &renderPassColorAttachment;
        renderPassDescriptor.depthStencilAttachment = nullptr;
        renderPassDescriptor.occlusionQuerySet = nullptr;
        renderPassDescriptor.timestampWrites = nullptr;

        WGPURenderPassEncoder renderPassEncoder = wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPassDescriptor);

        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, renderPipeline);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 0, vertexBuffer, 0, vertices.size() * sizeof(vertices[0]));
        wgpuRenderPassEncoderDraw(renderPassEncoder, 3, 1, 0, 0);

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
