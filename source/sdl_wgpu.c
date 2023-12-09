#include <webgpu-demo/sdl_wgpu.h>

#ifdef __APPLE__
#include <SDL_syswm.h>
#else
#include <SDL2/SDL_syswm.h>
#endif

WGPUSurface SDL_WGPU_CreateSurface(WGPUInstance instance, SDL_Window * window)
{
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    SDL_GetWindowWMInfo(window, &info);

    switch (info.subsystem)
    {
#ifdef SDL_VIDEO_DRIVER_X11
    case SDL_SYSWM_X11:
        {
            WGPUSurfaceDescriptorFromXlibWindow surfaceDescriptorFromXlibWindow;
            surfaceDescriptorFromXlibWindow.chain.next = 0;
            surfaceDescriptorFromXlibWindow.chain.sType = WGPUSType_SurfaceDescriptorFromXlibWindow;
            surfaceDescriptorFromXlibWindow.window = info.info.x11.window;
            surfaceDescriptorFromXlibWindow.display = info.info.x11.display;

            WGPUSurfaceDescriptor surfaceDescriptor;
            surfaceDescriptor.label = 0;
            surfaceDescriptor.nextInChain = (const WGPUChainedStruct*)&surfaceDescriptorFromXlibWindow;

            return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
        }
#endif

#ifdef SDL_VIDEO_DRIVER_WAYLAND
    case SDL_SYSWM_WAYLAND:
        {
            WGPUSurfaceDescriptorFromWaylandSurface surfaceDescriptorFromWaylandSurface;
            surfaceDescriptorFromWaylandSurface.chain.next = 0;
            surfaceDescriptorFromWaylandSurface.chain.sType = WGPUSType_SurfaceDescriptorFromWaylandSurface;
            surfaceDescriptorFromWaylandSurface.display = info.info.wl.display;
            surfaceDescriptorFromWaylandSurface.surface = info.info.wl.surface;

            WGPUSurfaceDescriptor surfaceDescriptor;
            surfaceDescriptor.label = 0;
            surfaceDescriptor.nextInChain = (const WGPUChainedStruct*)&surfaceDescriptorFromWaylandSurface;

            return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
        }
#endif

#ifdef SDL_VIDEO_DRIVER_WINDOWS
    case SDL_SYSWM_WINDOWS:
        {
            WGPUSurfaceDescriptorFromWindowsHWND surfaceDescriptorFromWindowsHWND;
            surfaceDescriptorFromWindowsHWND.chain.next = 0;
            surfaceDescriptorFromWindowsHWND.chain.sType = WGPUSType_SurfaceDescriptorFromWindowsHWND;
            surfaceDescriptorFromWindowsHWND.hwnd = info.info.win.window;
            surfaceDescriptorFromWindowsHWND.hinstance = info.info.win.hinstance;

            WGPUSurfaceDescriptor surfaceDescriptor;
            surfaceDescriptor.label = 0;
            surfaceDescriptor.nextInChain = (const WGPUChainedStruct*)&surfaceDescriptorFromWindowsHWND;

            return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
        }
#endif

        // TODO: support MacOS
    default:
        return 0;
    }

}
