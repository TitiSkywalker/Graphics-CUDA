/* A bunch of helper functions:
* 1. checked CUDA memory operations(malloc, memcpy, free)
* 2. memory allocations for user-defined classes:
*	i. allocate a piece of memory in GPU
*	ii. pass the pointer to kernel function, call "new" on this piece of memory
*	iii. "replacement new" will construct the object for you
*	(cannot directly copy an object from CPU to GPU because it won't preserve vtable)
* problem: too many synchronizations, this may slow down program if number of objects is large
*/
#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "Vecmath.cuh"
#include "Camera.cuh"
#include "Material.cuh"

#include "Light.cuh"

#include "Object3D.cuh"
#include "Sphere.cuh"
#include "Plane.cuh"
#include "Triangle.cuh"
#include "Velocity.cuh"

using namespace std;

//camera
__global__ static void allocatePerspectiveCamera(PerspectiveCamera* pointer, Vector3f center, Vector3f direction, Vector3f vertical, float angle)
{
	new(pointer) PerspectiveCamera(center, direction, vertical, angle);
}
//material
__global__ static void allocatePhong(Phong* pointer, Vector3f d_color, Vector3f s_color = Vector3f(0, 0, 0),
	float shine = 0, float rfrctIdx = 0)
{
	new(pointer) Phong(d_color, s_color, shine, rfrctIdx);
}
__global__ static void allocateAmbient(Ambient* pointer, Vector3f d_color, Vector3f s_color, float shine)
{
	new(pointer) Ambient(d_color, s_color, shine);
}
__global__ static void allocateGlossy(Glossy* pointer, Vector3f d_color, Vector3f s_color, float shine, float roughness)
{
	new(pointer) Glossy(d_color, s_color, shine, roughness);
}
__global__ static void allocateMirror(Mirror* pointer)
{
	new(pointer) Mirror();
}
__global__ static void allocateBackground(Background* pointer, Vector3f d_color)
{
	new(pointer) Background(d_color);
}
__global__ static void allocateEmit(Emit* pointer, Vector3f color, float falloff)
{
	new(pointer) Emit(color, falloff);
}
//light 
__global__ static void allocateDirectionalLight(DirectionalLight* pointer, Vector3f direction, Vector3f color)
{
	new(pointer) DirectionalLight(direction, color);
}
__global__ static void allocatePointLight(PointLight* pointer, Vector3f center, Vector3f color, float falloff)
{
	new(pointer) PointLight(center, color, falloff);
}
//object
__global__ static void allocateSphere(Sphere* pointer, Vector3f center, float radius, Material* material)
{
	new(pointer) Sphere(center, radius, material);
}
__global__ static void allocatePlane(Plane* pointer, Vector3f normal, float d, Material* material)
{
	new(pointer) Plane(normal, d, material);
}
__global__ static void allocateTriangle(Triangle* pointer, Vector3f a, Vector3f b, Vector3f c, Material* material)
{
	new(pointer) Triangle(a, b, c, material);
}
__global__ static void allocateVelocity(Velocity* pointer, Vector3f velocity, Object3D* object, Material* material)
{
	new(pointer) Velocity(velocity, object, material);
}

class Tool
{
public:
	//pass location string to these functions, and you will get more detailed error messages
	static void cudaMallocChecked(void** address, int size, string location = "???")
	{
		auto error = cudaMalloc(address, size);
		if (error != cudaSuccess)
		{
			cout << "Error: GPU memory allocation failed at " << location << endl;
			cout << "Message: " << cudaGetErrorString(error) << endl;
		}
	}

	static void cudaFreeChecked(void* address, string location = "???")
	{
		auto error = cudaFree(address);
		if (error != cudaSuccess)
		{
			cout << "Error: GPU memory release failed at " << location << endl;
			cout << "Message: " << cudaGetErrorString(error) << endl;
		}
	}

	static void cudaMemcpyChecked(void* destination, void* source, int size, cudaMemcpyKind flag, string location = "???")
	{
		auto error = cudaMemcpy(destination, source, size, flag);
		if (error != cudaSuccess)
		{
			cout << "Error: memory copy failed at " << location << endl;
			cout << "Message: " << cudaGetErrorString(error) << endl;
		}
	}

	static void deviceSynchronize(string location="???")
	{
		auto error = cudaDeviceSynchronize();
		if (error != cudaSuccess)
		{
			cout << "Error: GPU device syncronize failed at " << location << endl;
			cout << "Message: " << cudaGetErrorString(error) << endl;
		}
	}

	//camera allocator
	static PerspectiveCamera* newPerspectiveCamera(const Vector3f& center, const Vector3f& direction, const Vector3f& vertical, float angle)
	{
		PerspectiveCamera* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(PerspectiveCamera));
		allocatePerspectiveCamera << <1, 1 >> > (pointer, center, direction, vertical, angle);
		return pointer;
	}
	
	//material allocator
	static Phong* newPhong(const Vector3f& d_color, const Vector3f& s_color = Vector3f(0, 0, 0), float shine = 0, float rfrctIdx = 0)
	{
		Phong* pointer;
		cudaMallocChecked((void**) & pointer, sizeof(Phong));
		allocatePhong << <1, 1 >> > (pointer, d_color, s_color, shine, rfrctIdx);
		return pointer;
	}
	static Ambient* newAmbient(const Vector3f& d_color, const Vector3f& s_color = Vector3f(0, 0, 0), float shine = 0)
	{
		Ambient* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Ambient));
		allocateAmbient << <1, 1 >> > (pointer, d_color, s_color, shine);
		return pointer;
	}
	static Glossy* newGlossy(const Vector3f d_color, const Vector3f& s_color = Vector3f(0, 0, 0), float shine = 0, float roughness = 0.5)
	{
		Glossy* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Glossy));
		allocateGlossy << <1, 1 >> > (pointer, d_color, s_color, shine, roughness);
		return pointer;
	}
	static Mirror* newMirror()
	{
		Mirror* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Mirror));
		allocateMirror << <1, 1 >> > (pointer);
		return pointer;
	}
	static Background* newBackground(const Vector3f& d_color)
	{
		Background* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Background));
		allocateBackground << <1, 1 >> > (pointer, d_color);
		return pointer;
	}
	static Emit* newEmit(const Vector3f& color, float falloff)
	{
		Emit* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Emit));
		allocateEmit << <1, 1 >> > (pointer, color, falloff);
		return pointer;
	}

	//object allocator
	static Sphere* newSphere(const Vector3f& center, float radius, Material* material)
	{
		Sphere* pointer;
		cudaMallocChecked((void**) & pointer, sizeof(Sphere));
		allocateSphere << <1, 1 >> > (pointer, center, radius, material);
		return pointer;
	}
	static Plane* newPlane(const Vector3f& normal, float d, Material* material)
	{
		Plane* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Plane));
		allocatePlane << <1, 1 >> > (pointer, normal, d, material);
		return pointer;
	}
	static Triangle* newTriangle(const Vector3f& a, const Vector3f& b, const Vector3f& c, Material* m)
	{
		Triangle* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Triangle));
		allocateTriangle << <1, 1 >> > (pointer, a, b, c, m);
		return pointer;
	}
	static Velocity* newVelocity(const Vector3f& velocity, Object3D* object, Material* material)
	{
		Velocity* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(Velocity));
		allocateVelocity << <1, 1 >> > (pointer, velocity, object, material);
		return pointer;
	}

	//light allocator
	static DirectionalLight* newDirectionalLight(const Vector3f& d, const Vector3f& c)
	{
		DirectionalLight* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(DirectionalLight));
		allocateDirectionalLight << <1, 1 >> > (pointer, d, c);
		return pointer;
	}
	static PointLight* newPointLight(const Vector3f& p, const Vector3f& c, float fall)
	{
		PointLight* pointer;
		cudaMallocChecked((void**)&pointer, sizeof(PointLight));
		allocatePointLight << <1, 1 >> > (pointer, p, c, fall);
		return pointer;
	}
};