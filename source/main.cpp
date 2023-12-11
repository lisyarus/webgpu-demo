#include <webgpu.h>

#include <webgpu-demo/application.hpp>
#include <webgpu-demo/camera.hpp>
#include <webgpu-demo/engine.hpp>

#include <iostream>
#include <cstdint>
#include <chrono>
#include <unordered_set>


int main()
{
    Application application;
    Engine engine(application.device(), application.queue());
    auto renderObjects = engine.loadGLTF(PROJECT_ROOT "/Sponza/Sponza.gltf");

    auto sceneBbox = engine.bbox(renderObjects);

    Camera camera;
    camera.move((sceneBbox.first + sceneBbox.second) / 2.f);
    camera.setFov(glm::radians(45.f), application.width() * 1.f / application.height());
    float sceneDiagonal = glm::distance(sceneBbox.first, sceneBbox.second);
    camera.setClip(sceneDiagonal / 1000.f, sceneDiagonal);
    camera.setSpeed(sceneDiagonal / 10.f);

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

        auto surfaceTexture = application.nextSwapchainTexture();
        if (!surfaceTexture)
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
            .movingFast     = keysDown.contains(SDL_SCANCODE_LSHIFT),
            .movingSlow     = keysDown.contains(SDL_SCANCODE_LCTRL),
        });

        engine.render(surfaceTexture, renderObjects, camera);

        application.present();

        ++frameId;
    }
}
