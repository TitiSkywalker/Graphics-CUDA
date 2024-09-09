/* Triangle class is a light object class.
* Sampling a uniform triangle can be done by sampling a uniform square, then fold it:
*   ^ y
*   |
*   |\ - - - - 
*   |+ \ - - -
*   |+ + \ - -
*   |+ + + \ -
*   |+ + + + \
*   ------------------> x
*	(sampling both + and - is easy, then transform - into +)
*/
#pragma once
#include <cuda_runtime.h>
#include "Object3D.cuh"

using namespace std;

class Triangle : public Object3D
{
	friend class Velocity;

	Vector3f vertices[3];
	Vector3f normal;

public:
	__device__ Triangle(const Vector3f& a, const Vector3f& b, const Vector3f& c, Material* m) : Object3D(m)
	{
		vertices[0] = a;
		vertices[1] = b;
		vertices[2] = c;

		//counter clockwise
		normal = Vector3f::cross(b - a, c - b).normalized();
	}

	__device__ Triangle()
	{}

	__device__ object_type getType() const override
	{
		return TRIANGLE;
	}

	__device__ virtual bool intersect(const Ray& ray, Hit& hit, float tmin, curandState* state) const override
	{
		Vector3f a = vertices[0];
		Vector3f b = vertices[1];
		Vector3f c = vertices[2];

		Vector3f D1 = a - b;
		Vector3f D2 = a - c;
		Vector3f D3 = ray.getDirection();

		Matrix3f M(D1, D2, D3, true);

		bool issingular;
		Matrix3f MI = M.inverse(&issingular);
		if (issingular)
		{
			return false;
		}
		else
		{
			//solve for barycentric coordinates
			Vector3f B = a - ray.getOrigin();
			Vector3f x = MI * B;

			float beta = x[0];
			float gamma = x[1];
			float alpha = 1.0 - beta - gamma;
			float t = x[2];

			if ((alpha >= 0) && (beta >= 0) && (gamma >= 0) && (t > tmin) && (t < hit.getT()))
			{
				hit.set(t, material, normal);
				return true;
			}
			else
			{
				return false;
			}
		}
	}

	//randomly sample a position from light source
	__device__ virtual void getIllumination(const Vector3f& p, Vector3f& dir2Light, Vector3f& col, float& distanceToLight, curandState* state) const override
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int idy = threadIdx.y + blockIdx.y * blockDim.y;
		int id = idx + idy * blockDim.x * gridDim.x;

		//copy state to local memory for efficiency
		curandState localState = state[id];
		
		//get a uniform distribution on triangle region 
		float random1 = curand_uniform(&localState);
		float random2 = curand_uniform(&localState);
		if (random1 + random2 > 1.0)
		{
			//map to triangle: x1 = 1 - y0, y1 = 1 - x0
			float tmp = random1;
			random1 = 1 - random2;
			random2 = 1 - tmp;
		}

		//copy state back to global memory
		state[id] = localState;

		Vector3f a = vertices[0];
		Vector3f b = vertices[1];
		Vector3f c = vertices[2];

		Vector3f rand_position = a + random1 * (b - a) + random2 * (c - a);

		dir2Light = (rand_position - p);
		distanceToLight = dir2Light.length();
		dir2Light = dir2Light / distanceToLight;

		Vector3f color = material->getDiffuseColor();
		float falloff = ((Emit*)material)->falloff;
		col = color / (1 + falloff * distanceToLight * distanceToLight);
	}
};