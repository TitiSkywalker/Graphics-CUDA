/*Basic light object(point light& directional light).
* Instead of storing light information everywhere, just ask the light object.
*/
#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "Vector3f.cuh"
#include "Object3D.cuh"

using namespace std;

//I can't name the third variable as LIGHT because that name is already used in Material
enum light_type { DIRECTIONAL, POINT, BASE };   

//virtual class
class Light
{
public:
    CUDA_CALLABLE Light()
    {}

    CUDA_CALLABLE virtual ~Light()
    {}

    //calculate direction to light, light color and distance to light given observing point p
    CUDA_CALLABLE virtual void getIllumination(
        const Vector3f& p,
        Vector3f& dir,
        Vector3f& color,
        float& distanceToLight) const = 0;

    CUDA_CALLABLE virtual light_type getType() = 0;
};

class DirectionalLight : public Light
{
public:
    CUDA_CALLABLE DirectionalLight(const Vector3f& d, const Vector3f& c)
    {
        direction = d.normalized();
        color = c;
    }

    CUDA_CALLABLE ~DirectionalLight()
    {}

    CUDA_CALLABLE virtual void getIllumination(
        const Vector3f& p,
        Vector3f& dir2Light,
        Vector3f& col,
        float& distanceToLight) const override
    {
        dir2Light = -direction;
        col = color;
        distanceToLight = FLT_MAX;
    }

    CUDA_CALLABLE light_type getType()
    {
        return DIRECTIONAL;
    }

private:
    Vector3f direction;
    Vector3f color;
};

class PointLight : public Light
{
public:
    CUDA_CALLABLE PointLight(const Vector3f& p, const Vector3f& c, float fall)
    {
        position = p;
        color = c;
        falloff = fall;
    }

    CUDA_CALLABLE ~PointLight()
    {}

    CUDA_CALLABLE virtual void getIllumination(
        const Vector3f& p,
        Vector3f& dir2Light,
        Vector3f& col,
        float& distanceToLight) const override
    {
        dir2Light = (position - p);
        distanceToLight = dir2Light.length();
        dir2Light = dir2Light / distanceToLight;
        col = color / (1 + falloff * distanceToLight * distanceToLight);
    }

    CUDA_CALLABLE light_type getType()
    {
        return POINT;
    }

private:
    Vector3f position;
    Vector3f color;
    float falloff;
};
