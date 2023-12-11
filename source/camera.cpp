#include <webgpu-demo/camera.hpp>

#include <glm/gtx/transform.hpp>

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
    speed_ = speed;
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

    float speed = speed_;
    if (updateData.movingFast)
        speed *= 5.f;
    if (updateData.movingSlow)
        speed /= 20.f;

    if (updateData.movingForward)
        position_ += forward * speed * dt;
    if (updateData.movingBackward)
        position_ -= forward * speed * dt;
    if (updateData.movingLeft)
        position_ -= right * speed * dt;
    if (updateData.movingRight)
        position_ += right * speed * dt;
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
