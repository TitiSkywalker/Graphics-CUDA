/* Many material types, several notable things:
* 1. To avoid warp divergence in GPU, background(no hit) is treated as a special material.
*	 - This means that every ray will always "hit" something, even no object is hit.
*	 - To compute reflect ray, background does nothing, this ensures that the ray
*	   will keep hitting background since then.
*	 - To compute hitting point, it does not return infinity, but returns the origin.
*	 The above properties ensure that background is consistent with other material types.
* 2. Objects to not emit light, they are wrapped with a light emitting material.
*	 - The "getType" interface is used to recognize light objects.
*/
#pragma once
#include <curand_kernel.h>

#include "Vecmath.cuh"

#include "Ray.cuh"
#include "Hit.cuh"

enum material_type { PHONG, GLOSSY, MIRROR, AMBIENT, GLASS, BACKGROUND, EMIT, MATERIAL };

class Material
{
public:
	__device__ Material(
		const Vector3f& d_color,
		const Vector3f& s_color = Vector3f(0, 0, 0),
		float shine = 0, float rfrctIdx = 0, float rough = 0.5)
	{
		diffuseColor = d_color;
		specularColor = s_color;
		shininess = shine;
		refractionIndex = rfrctIdx;
		roughness = rough;
	}

	__device__ virtual ~Material()
	{}

	__device__ virtual float getRefractionIndex() const { return refractionIndex; }
	__device__ virtual float getRoughness() const { return roughness; }

	__device__ virtual Vector3f getDiffuseColor() const { return diffuseColor; }
	__device__ virtual Vector3f getSpecularColor() const { return specularColor; }

	__device__ virtual material_type getType() = 0;

	//default: Phong shading
	__device__ virtual Vector3f Shade(const Ray& ray, const Hit& hit, const Vector3f& dirToLight, const Vector3f& lightColor) const
	{
		//get local color
		Vector3f kd = diffuseColor;

		//compute local shading (without reflection)
		Vector3f n = hit.getNormal();

		Vector3f ks = specularColor;

		Vector3f reflection = 2 * (Vector3f::dot(dirToLight, n)) * n - dirToLight;
		float reflect = -Vector3f::dot(reflection, ray.getDirection());
		reflect = (reflect < 0) ? 0 : pow(reflect, shininess);

		Vector3f color = Vector3f::clampedDot(dirToLight, n) * Vector3f::pointwiseDot(lightColor, kd) +
			reflect * Vector3f::pointwiseDot(lightColor, ks);

		return color;
	}

	//default: perfect reflection
	__device__ virtual Ray computeReflection(const Ray& ray, const Hit& hit, curandState* state) const
	{
		Vector3f incoming = ray.getDirection();
		Vector3f normal = hit.getNormal();
		Vector3f reflected = (incoming - normal * 2 * Vector3f::dot(incoming, normal)).normalized();
		
		return Ray(ray.pointAtParameter(hit.getT()), reflected);
	}

	//default: point at parameter t
	__device__ virtual Vector3f computeHitPoint(const Ray& ray, const Hit& hit) const
	{
		return ray.pointAtParameter(hit.getT());
	}

	//default: simply add together
	__device__ virtual Vector3f blendColor(const Vector3f& localColor, const Vector3f& traceColor, const float fallOff,
		const Vector3f& normal, const Vector3f& incoming, const Vector3f& reflect)
	{
		return Vector3f::clamp(localColor + traceColor * fallOff);
	}

protected:
	Vector3f diffuseColor;
	Vector3f specularColor;
	float refractionIndex;
	float shininess;
	float roughness;
};

//simple, classic Phong material
class Phong : public Material
{
public:
	__device__ Phong(
		const Vector3f& d_color,
		const Vector3f& s_color = Vector3f(0, 0, 0),
		float shine = 0, float rfrctIdx = 0) :
		Material(d_color, s_color, shine, rfrctIdx , 0)
	{}

	__device__ ~Phong()
	{}

	__device__ material_type getType() override
	{
		return PHONG;
	}
};

//Mirror with perfect reflection
class Mirror : public Material
{
public:
	__device__ Mirror() : Material(Vector3f(0, 0, 0))
	{}

	__device__ ~Mirror()
	{}

	__device__ material_type getType() override
	{
		return MIRROR;
	}

	__device__ virtual Vector3f Shade(const Ray& ray, const Hit& hit, const Vector3f& dirToLight, const Vector3f& lightColor) const
	{
		return Vector3f(0, 0, 0);
	}

	//only look at reflect color
	__device__ virtual Vector3f blendColor(const Vector3f& localColor, const Vector3f& traceColor, const float fallOff,
		const Vector3f& normal, const Vector3f& incoming, const Vector3f& reflect)
	{
		return traceColor * fallOff;
	}
};

//ambient material with uniform BRDF (chalk, clay)
class Ambient : public Material
{
public:
	__device__ Ambient(const Vector3f& d_color, const Vector3f& s_color = Vector3f(0, 0, 0), float shine = 0) : Material(d_color, s_color, shine)
	{}

	__device__ ~Ambient()
	{}

	__device__ material_type getType() override
	{
		return AMBIENT;
	}

	//only shade diffuse color
	__device__ virtual Vector3f Shade(const Ray& ray, const Hit& hit, const Vector3f& dirToLight, const Vector3f& lightColor) const
	{
		//get local color
		Vector3f kd = diffuseColor;

		//compute local shading (without reflection)
		Vector3f n = hit.getNormal();

		Vector3f color = Vector3f::clampedDot(dirToLight, n) * Vector3f::pointwiseDot(lightColor, kd);

		return color;
	}

	//return a random reflect direction
	__device__ virtual Ray computeReflection(const Ray& ray, const Hit& hit, curandState* state) const override
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int idy = threadIdx.y + blockIdx.y * blockDim.y;
		int id = idx + idy * blockDim.x * gridDim.x;

		//copy state to local memory for efficiency
		curandState localState = state[id];
		//generate pseudo-random uniforms, curand_uniform generates number between 0 and 1
		float random1 = curand_uniform(&localState) - 0.5;
		float random2 = curand_uniform(&localState) - 0.5;
		float random3 = curand_uniform(&localState) - 0.5;
		//copy state back to global memory
		state[id] = localState;

		Vector3f reflectDir(random1, random2, random3);
		reflectDir.normalize();

		//make sure it points out
		Vector3f normal = hit.getNormal();
		if (Vector3f::dot(normal, reflectDir) < 0)
			reflectDir = -reflectDir;

		return Ray(ray.pointAtParameter(hit.getT()), reflectDir);
	}
};

//Glossy material with a loap BRDF (metal)
class Glossy : public Material
{
public:
	__device__ Glossy(
		const Vector3f& d_color,
		const Vector3f& s_color = Vector3f(0, 0, 0),
		float shine = 0, float rough = 0.5) :
		Material(d_color, s_color, shine, 0, rough)
	{}

	__device__ ~Glossy()
	{}

	__device__ material_type getType() override
	{
		return GLOSSY;
	}

	//Cook-Torrance BRDF function that takes roughness into account
	__device__ inline Vector3f CookTorrance(const Vector3f& normal, const Vector3f& incoming, const Vector3f& reflect)
	{
		Vector3f N = normal.normalized();
		Vector3f V = -incoming.normalized();
		Vector3f L = reflect.normalized();
		Vector3f H = (L + V).normalized();

		float H_N = Vector3f::dot(H, N);
		float H_V = Vector3f::dot(H, V);
		float N_L = Vector3f::dot(N, L);
		float N_V = Vector3f::dot(N, V);

		if (N_V == 0)
			return Vector3f(0, 0, 0);

		float delta = acos(H_N);    //angle between H and N
		float m = roughness;		//roughness
		float q = 5.0;              //specular reflection exponent

		float D = exp(-(tan(pow(delta / m, 2)))) / (m * m * pow(cos(delta), 4));
		float G = min(1.0, min(2.0 * H_N * N_V / H_V, 2.0 * H_N * N_L / H_V));
		float F = pow(H_N, q);

		float result = D * F * G / (N_L * N_V);

		if (isinf(result))
		{
			return Vector3f(1, 1, 1);
		}
		else if (isnan(result))
		{
			return Vector3f(0, 0, 0);
		}
		else if (result < 0)
		{
			result = -result;
		}

		return specularColor * result;
	}

	//only shade diffuse color
	__device__ virtual Vector3f Shade(const Ray& ray, const Hit& hit, const Vector3f& dirToLight, const Vector3f& lightColor) const
	{
		//get local color
		Vector3f kd = diffuseColor;

		//compute local shading (without reflection)
		Vector3f n = hit.getNormal();

		Vector3f color = Vector3f::clampedDot(dirToLight, n) * Vector3f::pointwiseDot(lightColor, kd);

		return color;
	}

	//BRDF function
	__device__ virtual Vector3f blendColor(const Vector3f& localColor, const Vector3f& traceColor, const float fallOff,
		const Vector3f& normal, const Vector3f& incoming, const Vector3f& reflect)
	{
		Vector3f brdf = CookTorrance(normal, incoming, reflect);

		return Vector3f::clamp(localColor + Vector3f::pointwiseDot(traceColor, brdf) * fallOff);
	}

	//return a random reflect direction
	__device__ virtual Ray computeReflection(const Ray& ray, const Hit& hit, curandState* state) const override
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int idy = threadIdx.y + blockIdx.y * blockDim.y;
		int id = idx + idy * blockDim.x * gridDim.x;

		//copy state to local memory for efficiency
		curandState localState = state[id];
		//generate pseudo-random uniforms, curand_uniform generates number between 0 and 1
		float random1 = curand_uniform(&localState) - 0.5;
		float random2 = curand_uniform(&localState) - 0.5;
		float random3 = curand_uniform(&localState) - 0.5;
		//copy state back to global memory
		state[id] = localState;

		Vector3f reflectDir(random1, random2, random3);
		reflectDir.normalize();

		//make sure it points out
		Vector3f normal = hit.getNormal();
		if (Vector3f::dot(normal, reflectDir) < 0)
			reflectDir = -reflectDir;

		return Ray(ray.pointAtParameter(hit.getT()), reflectDir);
	}
};

class Background : public Material
{
public:
	__device__ Background(const Vector3f& d_color) : Material(d_color)
	{}

	__device__ material_type getType() override
	{
		return BACKGROUND;
	}

	//return background color
	__device__ virtual Vector3f Shade(const Ray& ray, const Hit& hit, const Vector3f& dirToLight, const Vector3f& lightColor) const
	{
		return diffuseColor;
	}

	//no reflection
	__device__ virtual Ray computeReflection(const Ray& ray, const Hit& hit, curandState* state) const override
	{
		return ray;
	}

	//no hit point
	__device__ virtual Vector3f computeHitPoint(const Ray& ray, const Hit& hit) const override
	{
		return ray.getOrigin();
	}

	//no blending
	__device__ virtual Vector3f blendColor(const Vector3f& localColor, const Vector3f& traceColor, const float fallOff,
		const Vector3f& normal, const Vector3f& incoming, const Vector3f& reflect)
	{
		return localColor;
	}
};

class Emit : public Material
{
public:
	__device__ Emit(const Vector3f& color, float f) : Material(color), falloff(f)
	{}

	__device__ material_type getType()
	{
		return EMIT;
	}

	//emit color only
	__device__ virtual Vector3f Shade(const Ray& ray, const Hit& hit, const Vector3f& dirToLight, const Vector3f& lightColor) const
	{
		return diffuseColor;
	}

	//return a random reflect direction
	__device__ virtual Ray computeReflection(const Ray& ray, const Hit& hit, curandState* state) const override
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int idy = threadIdx.y + blockIdx.y * blockDim.y;
		int id = idx + idy * blockDim.x * gridDim.x;

		//copy state to local memory for efficiency
		curandState localState = state[id];
		//generate pseudo-random uniforms, curand_uniform generates number between 0 and 1
		float random1 = curand_uniform(&localState) - 0.5;
		float random2 = curand_uniform(&localState) - 0.5;
		float random3 = curand_uniform(&localState) - 0.5;
		//copy state back to global memory
		state[id] = localState;

		Vector3f reflectDir(random1, random2, random3);
		reflectDir.normalize();

		//make sure it points out
		Vector3f normal = hit.getNormal();
		if (Vector3f::dot(normal, reflectDir) < 0)
			reflectDir = -reflectDir;

		return Ray(ray.pointAtParameter(hit.getT()), reflectDir);
	}

	float falloff = 0;
};