#include <SDL2/SDL.h>

#include <webgpu.h>

#include <webgpu-demo/sdl_wgpu.h>

#include <iostream>
#include <vector>
#include <chrono>

int main()
{
    SDL_Init(SDL_INIT_VIDEO);

    int width = 1024;
    int height = 768;

    auto window = SDL_CreateWindow("WebGPU Demo", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    WGPUInstanceDescriptor instanceDescriptor;
    instanceDescriptor.nextInChain = nullptr;
    WGPUInstance instance = wgpuCreateInstance(&instanceDescriptor);

    std::cout << "Instance: " << instance << std::endl;

    WGPUSurface surface = SDL_WGPU_CreateSurface(instance, window);

    std::cout << "Surface: " << surface << std::endl;

    WGPURequestAdapterOptions requestAdapterOptions;
    requestAdapterOptions.nextInChain = nullptr;
    requestAdapterOptions.compatibleSurface = surface;
    requestAdapterOptions.powerPreference = WGPUPowerPreference_HighPerformance;
    requestAdapterOptions.backendType = WGPUBackendType_Vulkan;
    requestAdapterOptions.forceFallbackAdapter = false;

    WGPUAdapter adapter = nullptr;

    auto requestAdapterCallback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const * message, void * userdata){
        if (message)
            std::cout << "Adapter callback message: " << message << std::endl;

        if (status == WGPURequestAdapterStatus_Success)
            *(WGPUAdapter *)(userdata) = adapter;
        else
            throw std::runtime_error(message);
    };

    wgpuInstanceRequestAdapter(instance, &requestAdapterOptions, requestAdapterCallback, &adapter);

    std::cout << "Adapter: " << adapter << std::endl;

    if (!adapter)
        throw std::runtime_error("Adapter not created");

    std::vector<WGPUFeatureName> features;
    features.resize(wgpuAdapterEnumerateFeatures(adapter, nullptr));
    wgpuAdapterEnumerateFeatures(adapter, features.data());

    std::cout << "Adapter features:" << std::endl;
    for (auto const & feature : features)
        std::cout << "    " << feature << std::endl;

    WGPUSupportedLimits supportedLimits;
    supportedLimits.nextInChain = nullptr;
    wgpuAdapterGetLimits(adapter, &supportedLimits);

    std::cout << "Supported limits:" << std::endl;
    std::cout << "    maxTextureDimension1D: " << supportedLimits.limits.maxTextureDimension1D << std::endl;
    std::cout << "    maxTextureDimension2D: " << supportedLimits.limits.maxTextureDimension2D << std::endl;
    std::cout << "    maxTextureDimension3D: " << supportedLimits.limits.maxTextureDimension3D << std::endl;
    std::cout << "    maxTextureArrayLayers: " << supportedLimits.limits.maxTextureArrayLayers << std::endl;
    std::cout << "    maxBindGroups: " << supportedLimits.limits.maxBindGroups << std::endl;
    std::cout << "    maxBindGroupsPlusVertexBuffers: " << supportedLimits.limits.maxBindGroupsPlusVertexBuffers << std::endl;
    std::cout << "    maxBindingsPerBindGroup: " << supportedLimits.limits.maxBindingsPerBindGroup << std::endl;
    std::cout << "    maxDynamicUniformBuffersPerPipelineLayout: " << supportedLimits.limits.maxDynamicUniformBuffersPerPipelineLayout << std::endl;
    std::cout << "    maxDynamicStorageBuffersPerPipelineLayout: " << supportedLimits.limits.maxDynamicStorageBuffersPerPipelineLayout << std::endl;
    std::cout << "    maxSampledTexturesPerShaderStage: " << supportedLimits.limits.maxSampledTexturesPerShaderStage << std::endl;
    std::cout << "    maxSamplersPerShaderStage: " << supportedLimits.limits.maxSamplersPerShaderStage << std::endl;
    std::cout << "    maxStorageBuffersPerShaderStage: " << supportedLimits.limits.maxStorageBuffersPerShaderStage << std::endl;
    std::cout << "    maxStorageTexturesPerShaderStage: " << supportedLimits.limits.maxStorageTexturesPerShaderStage << std::endl;
    std::cout << "    maxUniformBuffersPerShaderStage: " << supportedLimits.limits.maxUniformBuffersPerShaderStage << std::endl;
    std::cout << "    maxUniformBufferBindingSize: " << supportedLimits.limits.maxUniformBufferBindingSize << std::endl;
    std::cout << "    maxStorageBufferBindingSize: " << supportedLimits.limits.maxStorageBufferBindingSize << std::endl;
    std::cout << "    minUniformBufferOffsetAlignment: " << supportedLimits.limits.minUniformBufferOffsetAlignment << std::endl;
    std::cout << "    minStorageBufferOffsetAlignment: " << supportedLimits.limits.minStorageBufferOffsetAlignment << std::endl;
    std::cout << "    maxVertexBuffers: " << supportedLimits.limits.maxVertexBuffers << std::endl;
    std::cout << "    maxBufferSize: " << supportedLimits.limits.maxBufferSize << std::endl;
    std::cout << "    maxVertexAttributes: " << supportedLimits.limits.maxVertexAttributes << std::endl;
    std::cout << "    maxVertexBufferArrayStride: " << supportedLimits.limits.maxVertexBufferArrayStride << std::endl;
    std::cout << "    maxInterStageShaderComponents: " << supportedLimits.limits.maxInterStageShaderComponents << std::endl;
    std::cout << "    maxInterStageShaderVariables: " << supportedLimits.limits.maxInterStageShaderVariables << std::endl;
    std::cout << "    maxColorAttachments: " << supportedLimits.limits.maxColorAttachments << std::endl;
    std::cout << "    maxColorAttachmentBytesPerSample: " << supportedLimits.limits.maxColorAttachmentBytesPerSample << std::endl;
    std::cout << "    maxComputeWorkgroupStorageSize: " << supportedLimits.limits.maxComputeWorkgroupStorageSize << std::endl;
    std::cout << "    maxComputeInvocationsPerWorkgroup: " << supportedLimits.limits.maxComputeInvocationsPerWorkgroup << std::endl;
    std::cout << "    maxComputeWorkgroupSizeX: " << supportedLimits.limits.maxComputeWorkgroupSizeX << std::endl;
    std::cout << "    maxComputeWorkgroupSizeY: " << supportedLimits.limits.maxComputeWorkgroupSizeY << std::endl;
    std::cout << "    maxComputeWorkgroupSizeZ: " << supportedLimits.limits.maxComputeWorkgroupSizeZ << std::endl;
    std::cout << "    maxComputeWorkgroupsPerDimension: " << supportedLimits.limits.maxComputeWorkgroupsPerDimension << std::endl;

    WGPURequiredLimits requiredLimits;
    requiredLimits.nextInChain = nullptr;
    requiredLimits.limits = supportedLimits.limits;

    WGPUDeviceDescriptor deviceDescriptor;
    deviceDescriptor.nextInChain = nullptr;
    deviceDescriptor.label = nullptr;
    deviceDescriptor.requiredFeatureCount = 0;
    deviceDescriptor.requiredFeatures = nullptr;
    deviceDescriptor.requiredLimits = &requiredLimits;
    deviceDescriptor.defaultQueue.nextInChain = nullptr;
    deviceDescriptor.defaultQueue.label = nullptr;
    deviceDescriptor.deviceLostCallback = nullptr;
    deviceDescriptor.deviceLostUserdata = nullptr;

    WGPUDevice device;

    auto requestDeviceCallback = [](WGPURequestDeviceStatus status, WGPUDevice device, char const * message, void * userdata)
    {
        if (message)
            std::cout << "Device callback message: " << message << std::endl;

        if (status == WGPURequestDeviceStatus_Success)
            *(WGPUDevice *)(userdata) = device;
        else
            throw std::runtime_error(message);
    };

    wgpuAdapterRequestDevice(adapter, &deviceDescriptor, requestDeviceCallback, &device);

    std::cout << "Device: " << device << std::endl;

    if (!device)
        throw std::runtime_error("Device not created");

    auto surfaceFormat = wgpuSurfaceGetPreferredFormat(surface, adapter);

    std::cout << "Surface format: " << surfaceFormat << std::endl;

    WGPUSurfaceConfiguration surfaceConfiguration;
    surfaceConfiguration.nextInChain = nullptr;
    surfaceConfiguration.device = device;
    surfaceConfiguration.format = surfaceFormat;
    surfaceConfiguration.usage = WGPUTextureUsage_RenderAttachment;
    surfaceConfiguration.viewFormatCount = 1;
    surfaceConfiguration.viewFormats = &surfaceFormat;
    surfaceConfiguration.alphaMode = WGPUCompositeAlphaMode_Auto;
    surfaceConfiguration.width = width;
    surfaceConfiguration.height = height;
    surfaceConfiguration.presentMode = WGPUPresentMode_Fifo;

    wgpuSurfaceConfigure(surface, &surfaceConfiguration);

    WGPUQueue queue = wgpuDeviceGetQueue(device);

    int frame = 0;

    for (bool running = true; running;)
    {
        std::cout << "Frame " << frame << std::endl;

        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        }

        WGPUSurfaceTexture surfaceTexture;

        auto getTextureStart = std::chrono::high_resolution_clock::now();

        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);

        if (surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated)
        {
            SDL_GetWindowSize(window, &width, &height);
            std::cout << "Resized to " << width << "x" << height << std::endl;

            surfaceConfiguration.width = width;
            surfaceConfiguration.height = height;
            wgpuSurfaceConfigure(surface, &surfaceConfiguration);

            wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
        }

        if (surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Timeout)
        {
            std::cout << "Timeout" << std::endl;
            ++frame;
            continue;
        }

        auto getTextureEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Waited " << (std::chrono::duration_cast<std::chrono::duration<double>>(getTextureEnd - getTextureStart).count() * 1000.0) << " ms for the swapchain image" << std::endl;

        if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_Success)
            throw std::runtime_error("Can't get surface texture: " + std::to_string(surfaceTexture.status));

        WGPUTextureViewDescriptor textureViewDescriptor;
        textureViewDescriptor.nextInChain = nullptr;
        textureViewDescriptor.label = nullptr;
        textureViewDescriptor.format = surfaceFormat;
        textureViewDescriptor.dimension = WGPUTextureViewDimension_2D;
        textureViewDescriptor.baseMipLevel = 0;
        textureViewDescriptor.mipLevelCount = 1;
        textureViewDescriptor.baseArrayLayer = 0;
        textureViewDescriptor.arrayLayerCount = 1;
        textureViewDescriptor.aspect = WGPUTextureAspect_All;

        WGPUTextureView surfaceTextureView = wgpuTextureCreateView(surfaceTexture.texture, &textureViewDescriptor);

        WGPUCommandEncoderDescriptor commandEncoderDescriptor;
        commandEncoderDescriptor.nextInChain = nullptr;
        commandEncoderDescriptor.label = nullptr;

        WGPUCommandEncoder commandEncoder = wgpuDeviceCreateCommandEncoder(device, &commandEncoderDescriptor);

        WGPURenderPassColorAttachment renderPassColorAttachment;
        renderPassColorAttachment.nextInChain = nullptr;
        renderPassColorAttachment.view = surfaceTextureView;
        renderPassColorAttachment.resolveTarget = nullptr;
        renderPassColorAttachment.loadOp = WGPULoadOp_Clear;
        renderPassColorAttachment.storeOp = WGPUStoreOp_Store;
        renderPassColorAttachment.clearValue = {1.0, 0.0, 1.0, 1.0};

        WGPURenderPassDescriptor renderPassDescriptor;
        renderPassDescriptor.nextInChain = nullptr;
        renderPassDescriptor.label = nullptr;
        renderPassDescriptor.colorAttachmentCount = 1;
        renderPassDescriptor.colorAttachments = &renderPassColorAttachment;
        renderPassDescriptor.depthStencilAttachment = nullptr;
        WGPUQuerySet occlusionQuerySet = nullptr;
        renderPassDescriptor.timestampWrites = nullptr;

        WGPURenderPassEncoder renderPassEncoder = wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPassDescriptor);
        wgpuRenderPassEncoderEnd(renderPassEncoder);

        wgpuTextureViewRelease(surfaceTextureView);

        WGPUCommandBufferDescriptor commandBufferDescriptor;
        commandBufferDescriptor.nextInChain = nullptr;
        commandBufferDescriptor.label = nullptr;

        WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(commandEncoder, &commandBufferDescriptor);
        wgpuQueueSubmit(queue, 1, &commandBuffer);

        wgpuSurfacePresent(surface);

        ++frame;
    }
}
