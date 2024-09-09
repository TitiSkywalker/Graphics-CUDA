//Hit stores the normal, material and the parameter t for a hitting point
#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "Vecmath.cuh"
#include "Ray.cuh"

class Material;
class LightObject;

class Hit
{
public:
	CUDA_CALLABLE Hit()
	{
		material = NULL;
		t = 1e38f;
	}
	CUDA_CALLABLE Hit(float _t, Material* m, const Vector3f& n)
	{
		t = _t;
		material = m;
		normal = n;
	}
	CUDA_CALLABLE Hit(const Hit& h)
	{
		t = h.t;
		material = h.material;
		normal = h.normal;
	}

	//no need for special destruction function

	CUDA_CALLABLE float getT() const
	{
		return t;
	}

	CUDA_CALLABLE Material* getMaterial() const
	{
		return material;
	}

	CUDA_CALLABLE const Vector3f& getNormal() const
	{
		return normal;
	}

	CUDA_CALLABLE void set(float _t, Material* m, const Vector3f& n)
	{
		t = _t;
		material = m;
		normal = n;
	}

	CUDA_CALLABLE void set(const Hit& hit)
	{
		t = hit.t;
		material = hit.material;
		normal = hit.normal;
	}

	Vector3f normal;
	Material* material;		//material is shared by many hit points, no need to free it here
	float t;
};