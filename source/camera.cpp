#include <webgpu-demo/camera.hpp>

#include <glm/gtx/transform.hpp>

#include <algorithm>

void Camera::move(glm::vec3 const & position)
{
    position_ = position;
}

void Camera::setRotation(float angleX, float angleY)
{
    xAngleTarget_ = angleX;
    xAngle_ = angleX;
    yAngleTarget_ = angleY;
    yAngle_ = angleY;
}

void Camera::setFov(float fovY, float aspectRatio)
{
    fovY_ = fovY;
    aspectRatio_ = aspectRatio;
}

void Camera::setClip(float near, float far)
{
    near_ = near;
    far_ = far;
}

void Camera::setSpeed(float speed)
{
    baseSpeed_ = speed;
}

void Camera::rotate(float deltaX, float deltaY)
{
    xAngleTarget_ += deltaX * sensitivity_;
    yAngleTarget_ += deltaY * sensitivity_;

    yAngleTarget_ = std::max(yAngleTarget_, -glm::pi<float>() / 2.f);
    yAngleTarget_ = std::min(yAngleTarget_,  glm::pi<float>() / 2.f);
}

void Camera::update(float dt, UpdateData const & updateData)
{
    float const rotationSmoothnessFactor = - std::expm1(- dt / rotationSmoothness_);

    xAngle_ += (xAngleTarget_ - xAngle_) * rotationSmoothnessFactor;
    yAngle_ += (yAngleTarget_ - yAngle_) * rotationSmoothnessFactor;

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

    float speedTarget = baseSpeed_;
    if (updateData.movingFast)
        speedTarget *= 5.f;
    if (updateData.movingSlow)
        speedTarget /= 10.f;

    glm::vec3 velocityTarget{0.f};
    if (updateData.movingForward)
        velocityTarget += forward;
    if (updateData.movingBackward)
        velocityTarget -= forward;
    if (updateData.movingLeft)
        velocityTarget -= right;
    if (updateData.movingRight)
        velocityTarget += right;

    if (glm::length(velocityTarget) > 1e-6)
        velocityTarget = glm::normalize(velocityTarget) * speedTarget;

    float const velocitySmoothnessFactor = - std::expm1(- dt / velocitySmoothness_);
    currentVelocity_ += (velocityTarget - currentVelocity_) * velocitySmoothnessFactor;

    position_ += currentVelocity_ * dt;
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
