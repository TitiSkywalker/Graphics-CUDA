//Simple ray with an origin and a direction
#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include <cassert>
#include <iostream>
#include "Vector3f.cuh"

class Ray
{
public:
    CUDA_CALLABLE Ray() = delete;
    CUDA_CALLABLE Ray(const Vector3f& orig, const Vector3f& dir)
    {
        origin = orig;
        direction = dir;
    }

    CUDA_CALLABLE Ray(const Ray& r)
    {
        origin = r.origin;
        direction = r.direction;
    }

    CUDA_CALLABLE void set(const Ray& ray)
    {
        origin = ray.origin;
        direction = ray.direction;
    }

    CUDA_CALLABLE const Vector3f& getOrigin() const
    {
        return origin;
    }

    CUDA_CALLABLE const Vector3f& getDirection() const
    {
        return direction;
    }

    CUDA_CALLABLE Vector3f pointAtParameter(float t) const
    {
        return origin + direction * t;
    }

private:

    Vector3f origin;
    Vector3f direction;

};