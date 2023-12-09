#include <webgpu.h>

#include <webgpu-demo/sdl_wgpu.h>
#include <webgpu-demo/application.hpp>

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
    std::uint32_t color;
    glm::vec3 normal;
};

static const char shaderCode[] =
R"(

@group(0) @binding(0) var<uniform> viewProjection: mat4x4f;

struct VertexInput {
    @builtin(vertex_index) index : u32,
    @location(0) position : vec3f,
    @location(1) color : vec4f,
    @location(2) normal : vec3f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) color : vec4f,
    @location(1) normal : vec3f,
}

@vertex
fn vertexMain(in : VertexInput) -> VertexOutput {
    return VertexOutput(viewProjection * vec4f(in.position, 1.0), in.color, in.normal);
}

@fragment
fn fragmentMain(in : VertexOutput) -> @location(0) vec4f {
    return in.color * (0.5 + 0.5 * dot(normalize(in.normal), normalize(vec3f(1.0, 2.0, 3.0))));
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

    WGPUVertexAttribute attributes[3];
    attributes[0].format = WGPUVertexFormat_Float32x3;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Unorm8x4;
    attributes[1].offset = 12;
    attributes[1].shaderLocation = 1;
    attributes[2].format = WGPUVertexFormat_Float32x3;
    attributes[2].offset = 16;
    attributes[2].shaderLocation = 2;

    WGPUVertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.arrayStride = sizeof(Vertex);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexBufferLayout.attributeCount = 3;
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

    std::vector<Vertex> vertices;

    {
        float const X = 0.525731112119133606f;
        float const Z = 0.850650808352039932f;

        glm::vec3 const icosahedronVertices[12]
        {
            {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
            {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
            {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
        };

        std::uint32_t const icosahedronTriangles[20][3]
        {
            {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
            {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
            {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
            {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11}
        };

        for (auto const & triangle : icosahedronTriangles)
        {
            std::uint32_t color = 0xffff3fffu;

            glm::vec3 triangleVertices[3]
            {
                icosahedronVertices[triangle[0]],
                icosahedronVertices[triangle[2]],
                icosahedronVertices[triangle[1]],
            };

            glm::vec3 normal = glm::normalize(glm::cross(triangleVertices[1] - triangleVertices[0], triangleVertices[2] - triangleVertices[0]));

            vertices.push_back(Vertex{triangleVertices[0], color, normal});
            vertices.push_back(Vertex{triangleVertices[1], color, normal});
            vertices.push_back(Vertex{triangleVertices[2], color, normal});
        }
    }

    WGPUBufferDescriptor vertexBufferDescriptor;
    vertexBufferDescriptor.nextInChain = nullptr;
    vertexBufferDescriptor.label = nullptr;
    vertexBufferDescriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
    vertexBufferDescriptor.size = vertices.size() * sizeof(vertices[0]);
    vertexBufferDescriptor.mappedAtCreation = false;

    WGPUBuffer vertexBuffer = wgpuDeviceCreateBuffer(application.device(), &vertexBufferDescriptor);

    wgpuQueueWriteBuffer(application.queue(), vertexBuffer, 0, vertices.data(), vertices.size() * sizeof(vertices[0]));

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

    glm::vec3 cameraPosition = glm::vec3(0.0, 0.0, 5.0);
    float cameraXAngle = 0.f;
    float cameraYAngle = 0.f;
    float cameraXAngleTarget = 0.f;
    float cameraYAngleTarget = 0.f;

    float cameraSensitivity = 0.003f;
    float cameraSpeed = 3.f;

    std::unordered_set<SDL_Scancode> keysDown;

    int frameId = 0;
    float time = 0.f;

    auto lastFrameStart = std::chrono::high_resolution_clock::now();

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
        case SDL_MOUSEMOTION:
            cameraXAngleTarget += event->motion.xrel * cameraSensitivity;
            cameraYAngleTarget += event->motion.yrel * cameraSensitivity;
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

        cameraXAngle += (cameraXAngleTarget - cameraXAngle) * (1.f - std::exp(- dt * 20.f));
        cameraYAngle += (cameraYAngleTarget - cameraYAngle) * (1.f - std::exp(- dt * 20.f));

        glm::vec3 cameraForward = glm::vec3(
            std::cos(cameraYAngle) * std::sin(cameraXAngle), -std::sin(cameraYAngle), - std::cos(cameraYAngle) * std::cos(cameraXAngle)
        );

        glm::vec3 cameraRight = glm::vec3(
           std::cos(cameraXAngle), 0.f, std::sin(cameraXAngle)
       );

        if (keysDown.contains(SDL_SCANCODE_W))
            cameraPosition += cameraForward * cameraSpeed * dt;
        if (keysDown.contains(SDL_SCANCODE_S))
            cameraPosition -= cameraForward * cameraSpeed * dt;
        if (keysDown.contains(SDL_SCANCODE_A))
            cameraPosition -= cameraRight * cameraSpeed * dt;
        if (keysDown.contains(SDL_SCANCODE_D))
            cameraPosition += cameraRight * cameraSpeed * dt;

        glm::mat4 viewMatrix =
            glm::rotate(cameraYAngle, glm::vec3(1.0, 0.0, 0.0)) *
            glm::rotate(cameraXAngle, glm::vec3(0.0, 1.0, 0.0)) *
            glm::translate(-cameraPosition);
        glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.f), application.width() * 1.f / application.height(), 0.01f, 100.f);
        glm::mat4 viewProjectionMatrix = projectionMatrix * viewMatrix;

        wgpuQueueWriteBuffer(application.queue(), uniformBuffer, 0, &viewProjectionMatrix, 64);

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
        wgpuRenderPassEncoderDraw(renderPassEncoder, vertices.size(), 1, 0, 0);

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
