//Camera can generate a ray according to the pixel's position on canvas
#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "Ray.cuh"
#include "Vecmath.cuh"

using namespace std;

//virtual camera class
class Camera
{
public:
    CUDA_CALLABLE Camera(
        const Vector3f& center,
        const Vector3f& direction,
        const Vector3f& vertical)
    {
        this->center = center;
        this->direction = direction.normalized();
        this->vertical = vertical.normalized();
        this->horizontal = Vector3f::cross(this->direction, vertical).normalized();
    }

    CUDA_CALLABLE Camera(
        const Vector3f& center,
        const Vector3f& direction,
        const Vector3f& vertical,
        const Vector3f& horizontal)
    {
        this->center = center;
        this->direction = direction.normalized();
        this->vertical = vertical.normalized();
        this->horizontal = horizontal;
    }

    CUDA_CALLABLE Camera(const Camera& camera)
    {
        center = camera.center;
        direction = camera.direction;
        vertical = camera.vertical;
        horizontal = camera.horizontal;
    }

    CUDA_CALLABLE virtual Ray generateRay(int x, int y, int width, int height) const = 0;

    CUDA_CALLABLE void setCenter(const Vector3f& pos)
    {
        this->center = pos;
    }
    CUDA_CALLABLE Vector3f getCenter() const
    {
        return this->center;
    }

    CUDA_CALLABLE void setRotation(const Matrix3f& mat)
    {
        this->horizontal = mat.getCol(0);
        this->vertical = -mat.getCol(1);
        this->direction = mat.getCol(2);
    }
    CUDA_CALLABLE Matrix3f getRotation() const
    {
        return Matrix3f(this->horizontal, -this->vertical, this->direction);
    }

    CUDA_CALLABLE virtual float getTMin() const = 0;

protected:
    Vector3f center;
    Vector3f direction;
    Vector3f vertical;
    Vector3f horizontal;
};

class PerspectiveCamera : public Camera
{ 
    float perspect_angle;
public:
    CUDA_CALLABLE PerspectiveCamera(
        const Vector3f& center,
        const Vector3f& direction,
        const Vector3f& vertical,
        float angle) :
        Camera(center, direction, vertical), perspect_angle(angle)
    {}

    CUDA_CALLABLE PerspectiveCamera(const PerspectiveCamera& camera) : Camera(
        camera.center, camera.direction, camera.vertical, camera.horizontal)
    {
        perspect_angle = camera.perspect_angle;
    }

    CUDA_CALLABLE virtual Ray generateRay(int x, int y, int width, int height) const
    {
        float fx = height / (2.0f * tan(perspect_angle / 2.0f));
        float fy = fx;

        Vector3f view =
            ((x - width / 2.0f) / fx) * horizontal +
            ((y - height / 2.0f) / fy) * vertical +
            direction;
        view.normalize();
        return Ray(center, view);
    }

    CUDA_CALLABLE virtual float getTMin() const
    {
        return 0.0f;
    }
};