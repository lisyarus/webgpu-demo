#pragma once

#include <glm/glm.hpp>

struct Camera
{
    struct UpdateData
    {
        bool movingForward;
        bool movingBackward;
        bool movingLeft;
        bool movingRight;
        bool movingFast;
        bool movingSlow;
    };

    void setFov(float fovY, float aspectRatio);
    void rotate(float deltaX, float deltaY);
    void update(float dt, UpdateData const & updateData);

    glm::mat4 viewMatrix() const;
    glm::mat4 projectionMatrix() const;
    glm::mat4 viewProjectionMatrix() const;

private:
    glm::vec3 position_ = glm::vec3(0.0, 0.0, 5.0);
    float xAngle_ = 0.f;
    float yAngle_ = 0.f;
    float xAngleTarget_ = 0.f;
    float yAngleTarget_ = 0.f;

    float fovY_ = std::atan(1.f);
    float aspectRatio_ = 1.f;
    float near_ = 0.01f;
    float far_ = 100.f;

    float sensitivity_ = 0.003f;
    float speed_ = 5.f;
    float smoothness_ = 20.f;
};
