#include <webgpu-demo/camera.hpp>

#include <glm/gtx/transform.hpp>

void Camera::setFov(float fovY, float aspectRatio)
{
    fovY_ = fovY;
    aspectRatio_ = aspectRatio;
}

void Camera::rotate(float deltaX, float deltaY)
{
    xAngleTarget_ += deltaX * sensitivity_;
    yAngleTarget_ += deltaY * sensitivity_;
}

void Camera::update(float dt, UpdateData const & updateData)
{
    float const smoothnessFactor = - std::expm1(- dt * smoothness_);

    xAngle_ += (xAngleTarget_ - xAngle_) * smoothnessFactor;
    yAngle_ += (yAngleTarget_ - yAngle_) * smoothnessFactor;

    glm::vec3 const forward = glm::vec3(
          std::cos(yAngle_) * std::sin(xAngle_),
        - std::sin(yAngle_),
        - std::cos(yAngle_) * std::cos(xAngle_)
    );

    glm::vec3 const right = glm::vec3(
       std::cos(xAngle_),
       0.f,
       std::sin(xAngle_)
   );

    if (updateData.movingForward)
        position_ += forward * speed_ * dt;
    if (updateData.movingBackward)
        position_ -= forward * speed_ * dt;
    if (updateData.movingLeft)
        position_ -= right * speed_ * dt;
    if (updateData.movingRight)
        position_ += right * speed_ * dt;
}

glm::mat4 Camera::viewMatrix() const
{
    return
        glm::rotate(yAngle_, glm::vec3(1.f, 0.f, 0.f)) *
        glm::rotate(xAngle_, glm::vec3(0.f, 1.f, 0.f)) *
        glm::translate(- position_);
}

glm::mat4 Camera::projectionMatrix() const
{
    return glm::perspective(fovY_, aspectRatio_, near_, far_);
}

glm::mat4 Camera::viewProjectionMatrix() const
{
    return projectionMatrix() * viewMatrix();
}
