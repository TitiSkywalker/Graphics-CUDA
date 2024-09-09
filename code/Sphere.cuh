//Simple sphere class that cannot emit light
#pragma once
#include <cmath>

#include "Object3d.cuh"
#include <gsl/gsl_math.h>	//get M_PI

class Sphere : public Object3D
{
	friend class Velocity;

	Vector3f center;
	float radius;

public:
	__device__ Sphere(const Vector3f& c, float r, Material* material): Object3D(material)
	{
		center = c;
		radius = r;
	}

	__device__ Sphere(const Sphere& sphere)
	{
		center = sphere.center;
		radius = sphere.radius;
		material = sphere.material;
	}

	//solve a quadratic equation
	__device__ virtual bool intersect(const Ray& r, Hit& h, float tmin, curandState* state) const override
	{
		//a*t^2+2b*t+c=0
		float a = Vector3f::dot(r.getDirection(), r.getDirection());
		float b = Vector3f::dot(r.getOrigin() - center, r.getDirection());
		float c = Vector3f::dot(r.getOrigin() - center, r.getOrigin() - center) - radius * radius;

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

	__device__ virtual object_type getType() const override
	{
		return SPHERE;
	}
};