/* Monte Carlo path tracing implemented with CUDA kernel function.
* 1. Each thread computes color for one pixel, position is determined by threadIdx and blockIdx.
* 2. Stack space is limited, so I've changed the recursion into a loop.
* 3. To intersect objects and sample light sources, move them into shared memory
*/
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>

#include "Material.cuh"
#include "Object3D.cuh"
#include "Sphere.cuh"
#include "Plane.cuh"
#include "Triangle.cuh"

#include "SceneParser.cuh"

using namespace std;

//send a package to GPU consisting of information about scene
struct Package
{
	Vector3f ambientLight;		//background ambient light
	Background* background;		//background is treated as a special material
	Camera* camera;				//camera is located in global memory, move it into shared memory
	
	Object3D** objects;
	Object3D** lightObjects;
	Light** lights;

	curandState* randStates;	//CUDA pseudo random number generator
	Vector3f* image;			//store final color here

	int width;
	int height;
	int numObjects;
	int numLights;
	int numLightObjects;
	int sampleRate;
	int maxDepth;
};

//use union to automatically compute the size of shared classes
union SharedObject
{
	Sphere		object1;
	Plane		object2;
	Triangle	object3;
	Velocity	object4;

	~SharedObject() {}
};
union SharedLight
{
	DirectionalLight	light1;
	PointLight			light2;

	~SharedLight() {}
};

//use shared memory for object intersection and light shading
__shared__ char sharedObject[Configuration::BLOCKX][sizeof(SharedObject)];
__shared__ char sharedLight[Configuration::BLOCKX][sizeof(SharedLight)];
__shared__ char sharedLightObject[Configuration::BLOCKX][sizeof(SharedObject)];

//when number of objects is small, just copy once
__shared__ bool skipCopyObjects;
__shared__ bool skipCopyLights;
__shared__ bool skipCopyLightObjects;

//move objects
__device__ static inline void moveSharedGroup(const Package& package, int base)
{
	__syncthreads();
	if (threadIdx.y == 0) 
	{
		int index = threadIdx.x + base;
		Object3D* target = (index < package.numObjects) ? package.objects[index] : NULL;
		object_type type = (target) ? target->getType() : OBJECT;
		
		switch (type)
		{
			case SPHERE:
				new(sharedObject[threadIdx.x]) Sphere(*(Sphere*)target);
				break;
			case PLANE:
				new(sharedObject[threadIdx.x]) Plane(*(Plane*)target);
				break;
			case TRIANGLE:
				new(sharedObject[threadIdx.x]) Triangle(*(Triangle*)target);
				break;
			case VELOCITY:
				new(sharedObject[threadIdx.x]) Velocity(*(Velocity*)target);
				break;
		}
	}
	__syncthreads();
}

//move lights
__device__ static inline void moveSharedLights(const Package& package, int base)
{
	__syncthreads();
	if (threadIdx.y == 0)
	{
		int index = threadIdx.x + base;
		Light* target = (index < package.numLights) ? package.lights[index] : NULL;
		light_type lightType = (target) ? target->getType() : BASE;

		switch (lightType)
		{
		case DIRECTIONAL:
			new(sharedLight[threadIdx.x]) DirectionalLight(*(DirectionalLight*)target);
			break;
		case POINT:
			new(sharedLight[threadIdx.x]) PointLight(*(PointLight*)target);
			break;
		}
	}
	__syncthreads();
}

//move light objects
__device__ static inline void moveSharedLightGroup(const Package& package, int base)
{
	__syncthreads();
	if (threadIdx.y == 0)
	{
		int index = threadIdx.x + base;
		Object3D* target = (index < package.numLightObjects) ? package.lightObjects[index] : NULL;
		object_type type = (target) ? target->getType() : OBJECT;

		switch (type)
		{
		case SPHERE:
			new(sharedLightObject[threadIdx.x]) Sphere(*(Sphere*)target);
			break;
		case PLANE:
			new(sharedLightObject[threadIdx.x]) Plane(*(Plane*)target);
			break;
		case TRIANGLE:
			new(sharedLightObject[threadIdx.x]) Triangle(*(Triangle*)target);
		}
	}
	__syncthreads();
}

__device__ static inline bool intersectGroup(const Package& package, const Ray& ray, Hit& hit)
{
	int numObjects = package.numObjects;
	bool hasHit = false;
	for (int i = 0; i < numObjects; i += Configuration::BLOCKX)
	{
		if (!skipCopyObjects)
			moveSharedGroup(package, i);

		for (int j = 0; j < Configuration::BLOCKX && j + i < numObjects; j++)
		{
			hasHit |= ((Object3D*)sharedObject[j])->intersect(ray, hit, EPSILON, package.randStates);
		}
	}
	return hasHit;
}

__device__ static inline Vector3f renderLocalColor(const Package& package, const Ray& ray, const Hit& hit)
{
	Vector3f color;

	Material* material = hit.getMaterial();

	//in case of background, do not use ray.pointAtParameter
	Vector3f center = material->computeHitPoint(ray, hit);
	
	//render color for point and directional lights
	int numLights = package.numLights;
	for (int i = 0; i < numLights; i+=Configuration::BLOCKX)
	{
		if (!skipCopyLights)
			moveSharedLights(package, i);

		for (int j = 0; j < Configuration::BLOCKX && j + i < numLights; j++) 
		{
			Vector3f lightColor;
			Vector3f dir2light;
			float distance = 0;
			((Light*)sharedLight[j])->getIllumination(center, dir2light, lightColor, distance);

			//cast shadow rays fr all kinds of objects
			Ray shadowRay(center, dir2light);
			Hit shadowHit(1e38, package.background, Vector3f(0, 0, 0));
			bool hasHitGroup = intersectGroup(package, shadowRay, shadowHit);

			if (hasHitGroup && shadowHit.getT() < distance)
				continue;
			color += material->Shade(ray, hit, dir2light, lightColor);
		}
	}

	//render color for light objects
	int numLightObjects = package.numLightObjects;
	for (int i = 0; i < numLightObjects; i += Configuration::BLOCKX)
	{
		if (!skipCopyLightObjects)
			moveSharedLightGroup(package, i);

		for (int j = 0; j < Configuration::BLOCKX && j + i < numLightObjects; j++)
		{
			//sample point from light object
			Vector3f lightColor;
			Vector3f dir2light;
			float distance = 0;
			((Object3D*)sharedLightObject[j])->getIllumination(center, dir2light, lightColor, distance, package.randStates);

			//cast shadow rays, dir2light aready normalized
			Ray shadowRay(center, dir2light);
			Hit shadowHit(1e38, package.background, Vector3f(0, 0, 0));
			bool hasHitGroup = intersectGroup(package, shadowRay, shadowHit);

			//-¦Å to avoid self-shadowing, which will happen when t = distance
			if (hasHitGroup && shadowHit.getT() < distance - EPSILON)
				continue;
			color += material->Shade(ray, hit, dir2light, lightColor);
		}
	}
	return color;
}

//modify ray to be the next ray, compute hit point and local color, update intensity falloff
__device__ static inline Vector3f traceRay(const Package& package, Ray& ray, float& falloff, Hit& hit)
{
	bool group_intersect = intersectGroup(package, ray, hit);
	Material* material = hit.getMaterial();

	//may intersect a normal object or a light object
	Vector3f localColor = renderLocalColor(package, ray, hit);

	if (group_intersect)
	{
		//hit a light object
		if (material->getType() == EMIT)
			localColor = material->Shade(ray, hit, Vector3f(0, 0, 0), Vector3f(0, 0, 0));

		//compute next ray
		ray = material->computeReflection(ray, hit, package.randStates);

		//accumulate falloff
		float distance = hit.getT();
		falloff = falloff / (1 + FALLOFF * distance * distance);

		return localColor;
	}
	else
	{
		//return background color, future traces will be ignored 
		falloff = 0;
		return material->Shade(ray, hit, Vector3f(0, 0, 0), Vector3f(0, 0, 0));
	}
}

//Monte Carlo path tracing implemented with a loop
__global__ static void render(Package package)
{
	__shared__ char shared_camera[sizeof(PerspectiveCamera)];
	int ix = threadIdx.x;
	int iy = threadIdx.y;

	int x = blockIdx.x * blockDim.x + ix;
	int y = blockIdx.y * blockDim.y + iy;

	//copy camera into shared memory once for all
	if (ix == 0 && iy == 0)
	{
		PerspectiveCamera* camera = new(shared_camera) PerspectiveCamera(*(PerspectiveCamera*)package.camera);
	}
	__syncthreads();

	//when the number of objects is small, copy once is enough
	{
		if (package.numObjects <= Configuration::BLOCKX)
		{
			moveSharedGroup(package, 0);
			skipCopyObjects = true;
		}
		else if (ix == 0 && iy == 0)
			skipCopyObjects = false;

		if (package.numLights <= Configuration::BLOCKX)
		{
			moveSharedLights(package, 0);
			skipCopyLights = true;
		}
		else if (ix == 0 && iy == 0)
			skipCopyLights = false;

		if (package.numLightObjects <= Configuration::BLOCKX)
		{
			moveSharedLightGroup(package, 0);
			skipCopyLightObjects = true;
		}
		else if (ix == 0 && iy == 0)
			skipCopyLightObjects = false;
	}

	//avoid recomputing camera rays
	Ray baseRay = ((PerspectiveCamera*)shared_camera)->generateRay(x, y, package.width, package.height);

	Vector3f color;
	for (int i = 0; i < package.sampleRate; i++)
	{
		//"nextRay" is for future ray tracing
		Ray nextRay(baseRay);
		//these two rays are necessary for computing BRDF function
		Ray thisRay(baseRay);
		Ray prevRay(baseRay);

		float thisFallOff = 1;
		Hit thisHit(1e38, package.background, Vector3f(0, 0, 0));

		//compute local color for the first ray
		Vector3f localColor = traceRay(package, nextRay, thisFallOff, thisHit);
		Hit prevHit(thisHit);
		thisFallOff = 1;
		thisRay.set(nextRay);

		//change recursive ray tracing into a loop, reduce stack space
		for (int _ = 0; _ < package.maxDepth - 1; _++)
		{
			thisHit.set(1e38, package.background, Vector3f(0, 0, 0));
			//compute secondary color
			Vector3f traceColor = traceRay(package, nextRay, thisFallOff, thisHit);

			//blend color with local color
			localColor = prevHit.getMaterial()->blendColor(localColor, traceColor, thisFallOff,
				prevHit.getNormal(), prevRay.getDirection(), thisRay.getDirection());
			
			prevRay.set(thisRay);
			thisRay.set(nextRay);
			prevHit.set(thisHit);
		}
		color = color + localColor;
	}
	color = color / package.sampleRate;

	//(hopefully) memory colescing
	package.image[y * package.width + x] = Vector3f::clamp(color);
}

//set up random number seeds for CUDA pseudo-random generator
__global__ static void setupRandom(curandState* state, unsigned long seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	int id = idx + idy * blockDim.x * gridDim.x;

	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init(seed, id, 0, &state[id]);
}