/*Velocity assigns velocity to an object, can get motion blur effect.
* I forceinlined secondary intersect calls to save stack space
*/
#pragma once
#include <iostream>
#include <random>

#include "Vecmath.cuh"
#include "Object3D.cuh"

using namespace std;

class Velocity : public Object3D
{
	Object3D* object;
	Vector3f velocity;

	__device__ __forceinline__ bool intersectSphere(Sphere* sphere, const Ray& r, Hit& h, float tmin) const
	{
		//a*t^2+2b*t+c=0
		Vector3f direction = r.getDirection();
		Vector3f origin = r.getOrigin();
		Vector3f center = sphere->center;
		float radius = sphere->radius;
		float a = Vector3f::dot(direction, direction);
		float b = Vector3f::dot(origin - center, direction);
		float c = Vector3f::dot(origin - center, origin - center) - radius * radius;

		//delta=4b^2-4ac, quarter_delta=b^2-4ac
		float quarter_delta = b * b - a * c;
		if (quarter_delta < 0)
		{
			//no intersection
			return false;
		}
		else if (quarter_delta < 1e-20)
		{
			//roughly 1 intersection
			float t = -b / a;

			if (t > tmin && t < h.getT())
			{
				Vector3f normal = r.pointAtParameter(t) - center;
				normal.normalize();
				h.set(t, material, normal);
				return true;
			}
			return false;
		}
		else
		{
			//2 intersections
			float sqrt_delta = sqrt(quarter_delta);
			float t1 = (-b - sqrt_delta) / a;
			float t2 = (-b + sqrt_delta) / a;
			bool changed = false;
			if (t1 > tmin && t1 < h.getT())
			{
				Vector3f normal = r.pointAtParameter(t1) - center;
				normal.normalize();
				h.set(t1, material, normal);
				changed = true;
			}
			if (t2 > tmin && t2 < h.getT())
			{
				Vector3f normal = r.pointAtParameter(t2) - center;
				normal.normalize();
				h.set(t2, material, normal);
				changed = true;
			}
			return changed;
		}
	}

	__device__ __forceinline__ bool intersectPlane(Plane* plane, const Ray& r, Hit& h, float tmin) const
	{
		Vector3f N = plane->N;
		float D = plane->D;
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

	__device__ __forceinline__ bool intersectTriangle(Triangle* triangle, const Ray& r, Hit& h, float tmin) const
	{
		Vector3f* vertices = triangle->vertices;

		Vector3f a = vertices[0];
		Vector3f b = vertices[1];
		Vector3f c = vertices[2];

		Vector3f D1 = a - b;
		Vector3f D2 = a - c;
		Vector3f D3 = r.getDirection();

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
			Vector3f B = a - r.getOrigin();
			Vector3f x = MI * B;

			float beta = x[0];
			float gamma = x[1];
			float alpha = 1.0 - beta - gamma;
			float t = x[2];

			if ((alpha >= 0) && (beta >= 0) && (gamma >= 0) && (t > tmin) && (t < h.getT()))
			{
				h.set(t, material, triangle->normal);
				return true;
			}
			else
			{
				return false;
			}
		}
	}

public:
	__device__ Velocity(const Vector3f& v, Object3D* o, Material* m) : Object3D(m)
	{
		velocity = v;
		object = o;
	}

	__device__ virtual bool intersect(const Ray& r, Hit& h, float tmin, curandState* state) const override
	{
		//radomly sample a time between -1 and 1
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int idy = threadIdx.y + blockIdx.y * blockDim.y;
		int id = idx + idy * blockDim.x * gridDim.x;

		//copy state to local memory for efficiency
		curandState localState = state[id];

		//get a uniform distribution on triangle region 
		float time = curand_uniform(&localState);

		//copy state back to global memory
		state[id] = localState;

		Vector3f offset = velocity * time;

		//move this object is equal to moving the incoming ray at opposite direction
		Vector3f origin = r.getOrigin() - offset;
		Ray newRay(origin, r.getDirection());

		object_type type = object->getType();
		switch (type)
		{
			case SPHERE:	return intersectSphere((Sphere*)object, newRay, h, tmin);
			case TRIANGLE:	return intersectTriangle((Triangle*)object, newRay, h, tmin);
			case PLANE:		return intersectPlane((Plane*)object, newRay, h, tmin);
			default:		return false;
		}
	}

	__device__ object_type getType() const override
	{
		return VELOCITY;
	}
};