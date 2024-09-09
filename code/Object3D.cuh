/* Object3D is a virtual class:
* 1. It can compute intersection point with an incoming ray
* 2. For light objects, they need to sample on their surface
*/
#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Material.cuh"

enum object_type { TRIANGLE, SPHERE, GROUP, MESH, PLANE, TRANSFORM, VELOCITY, OBJECT };

class Object3D
{
public:
	__device__ Object3D()
	{
		material = NULL;
	}
	__device__ Object3D(Material* m)
	{
		material = m;
	}

	__device__ virtual ~Object3D() {}

	__device__ virtual bool intersect(const Ray& r, Hit& h, float tmin, curandState* state) const = 0;

	__device__ virtual void getIllumination(const Vector3f& p, Vector3f& dir2Light, Vector3f& col, float& distanceToLight, curandState* state) const
	{
		printf("Error: getIllumination not implemented yet\n");
	}

	__device__ virtual Material* getMaterial() const
	{
		return material;
	}

	__device__ virtual object_type getType() const
	{
		return OBJECT;
	}

protected:
	Material* material;
};