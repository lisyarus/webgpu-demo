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

    void move(glm::vec3 const & position);
    void setRotation(float angleX, float angleY);
    void setFov(float fovY, float aspectRatio);
    void setClip(float near, float far);
    void setSpeed(float speed);
    void rotate(float deltaX, float deltaY);
    void update(float dt, UpdateData const & updateData);

    glm::mat4 viewMatrix() const;
    glm::mat4 projectionMatrix() const;
    glm::mat4 viewProjectionMatrix() const;
    glm::vec3 position() const { return position_; }

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
    float baseSpeed_ = 5.f;
    glm::vec3 currentVelocity_ = glm::vec3(0.f);
    float rotationSmoothness_ = 0.05f;
    float velocitySmoothness_ = 0.08f;
};
