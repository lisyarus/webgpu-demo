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
    engine.setEnvMap(PROJECT_ROOT "/clarens_midday_4k.hdr");
    auto renderObjects = engine.loadGLTF(PROJECT_ROOT "/Sponza/Sponza.gltf");

    auto sceneBbox = engine.bbox(renderObjects);

    Camera camera;
    camera.move({-9.f, 1.5f, -0.25f});
//    camera.move({-8.51785f, 1.50002f, 0.910748f});
    camera.setRotation(glm::radians(90.f), 0.f);
//    camera.setRotation(0.889796f, 0.108f);
    camera.setFov(glm::radians(45.f), application.width() * 1.f / application.height());
    float sceneDiagonal = glm::distance(sceneBbox.min, sceneBbox.max);
    camera.setClip(sceneDiagonal / 1000.f, sceneDiagonal);
    camera.setSpeed(sceneDiagonal / 10.f);

    std::unordered_set<SDL_Scancode> keysDown;

    int frameId = 0;
    float time = 0.f;

    bool paused = true;
    bool day = true;
    bool vsync = true;

    auto lastFrameStart = std::chrono::high_resolution_clock::now();

    glm::vec3 shockCenter(0.f);
    float shockDistance = 1e9f;

    for (bool running = true; running;)
    {
//        std::cout << "Frame " << frameId << std::endl;

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
                application.resize(event->window.data1, event->window.data2, vsync);
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
            if (event->key.keysym.scancode == SDL_SCANCODE_SPACE)
                paused ^= true;
            if (event->key.keysym.scancode == SDL_SCANCODE_N)
            {
                day ^= true;

                if (day)
                    engine.setEnvMap(PROJECT_ROOT "/clarens_midday_4k.hdr");
                else
                    engine.setEnvMap(PROJECT_ROOT "/satara_night_4k.hdr");
            }
            if (event->key.keysym.scancode == SDL_SCANCODE_V)
            {
                vsync ^= true;
                application.resize(application.width(), application.height(), vsync);
            }
            if (event->key.keysym.scancode == SDL_SCANCODE_F)
            {
                shockCenter = camera.position();
                shockDistance = 0.f;
            }
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
        if (!paused)
        {
            time += dt;
            shockDistance += sceneDiagonal * 0.4f * dt;
        }
        lastFrameStart = thisFrameStart;

        camera.update(dt, {
            .movingForward  = keysDown.contains(SDL_SCANCODE_W),
            .movingBackward = keysDown.contains(SDL_SCANCODE_S),
            .movingLeft     = keysDown.contains(SDL_SCANCODE_A),
            .movingRight    = keysDown.contains(SDL_SCANCODE_D),
            .movingFast     = keysDown.contains(SDL_SCANCODE_LSHIFT),
            .movingSlow     = keysDown.contains(SDL_SCANCODE_LCTRL),
        });

        Engine::Settings settings;

        float const lightRotationSpeed = 0.1f;
        float const lightPhaseShift = 0.f;

        float const lightPhase = time * lightRotationSpeed + lightPhaseShift;

        if (day)
        {
            settings =
            {
                .skyColor = {0.4f, 0.7f, 1.f},
                .ambientLight = {0.5f, 0.4f, 0.3f},
                .envIntensity = 0.5f,
                .sunDirection = glm::normalize(glm::vec3{std::cos(lightPhase), 2.f, std::sin(lightPhase)}),
                .sunIntensity = {20.f, 16.f, 12.f},
            };
        }
        else
        {
            settings =
            {
                .skyColor = {0.0f, 0.0f, 0.001f},
                .ambientLight = {0.05f, 0.1f, 0.15f},
                .envIntensity = 0.05f,
                .sunDirection = glm::normalize(glm::vec3{std::cos(lightPhase), 2.f, std::sin(lightPhase)}),
                .sunIntensity = {1.f, 2.f, 3.f},
            };
        }

        settings.paused = paused;
        settings.shockCenter = shockCenter;
        settings.shockDistance = shockDistance;
        settings.dt = dt;

        engine.render(surfaceTexture, renderObjects, camera, sceneBbox, settings);

        application.present();

        ++frameId;
    }
}
