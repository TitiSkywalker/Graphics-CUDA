//Simple plane class that cannot emit light
#pragma once
#include "Object3d.cuh"
#include "Vecmath.cuh"
#include <cmath>

class Plane : public Object3D
{
    friend class Velocity;

    Vector3f N;
    float D;

public:
    __device__ Plane()
    {
        D = 0;
    }

    __device__ Plane(const Vector3f& normal, float d, Material* m): Object3D(m)
    {
        N = normal.normalized();
        D = -d;
    }

    __device__ bool intersect(const Ray& r, Hit& h, float tmin, curandState* state) const override
    {
        float parallel = Vector3f::dot(N, r.getDirection());
        if (parallel == 0)
        {
            return false;
        }
        float t = -(D + Vector3f::dot(N, r.getOrigin())) / parallel;

        if (t > tmin && t < h.getT())
        {
            if (parallel > 0)
                h.set(t, material, -N);
            else
                h.set(t, material, N);
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ object_type getType() const override
    {
        return PLANE;
    }
};