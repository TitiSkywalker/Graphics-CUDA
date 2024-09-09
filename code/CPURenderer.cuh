/* This is what we need to do in CPU side:
* 1. parse the scene file, pre-allocate objects in GPU global memory.
* 2. prepare a package with necessary information.
* 3. prepare GPU running environment(setup CUDA pseudo random number seed).
* 4. --- GPU ---
* 5. move image from GPU into CPU, perform anti-aliasing processes, finally save image.
*/
#pragma once
#include <iostream>

#include "Configuration.cuh"
#include "GPURenderer.cuh"
#include "Image.cuh"

using namespace std;

class Renderer
{
	//for jittored sampling, width & height is 3x larger
	static Package prepareScene(int width, int height, const SceneParser& sceneParser)
	{
		//prepare a package for GPU
		Package package;
		package.background = sceneParser.getBackground();
		package.ambientLight = sceneParser.getAmbientLight();
		package.camera = sceneParser.getCamera();
		package.width = width;
		package.height = height;
		package.numObjects = sceneParser.getNumObjects();
		package.numLights = sceneParser.getNumLights();
		package.numLightObjects = sceneParser.getNumLightObjects();
		package.sampleRate = Configuration::SAMPLERATE;
		package.maxDepth = Configuration::MAXDEPTH;

		//setup CUDA kernel for random numbers
		curandState* randStates;
		Tool::cudaMallocChecked((void**)&randStates, width * height * sizeof(curandState));
		int gridx = (width + Configuration::BLOCKX - 1) / Configuration::BLOCKX;
		int gridy = (height + Configuration::BLOCKY - 1) / Configuration::BLOCKY;
		dim3 grid(gridx, gridy);
		dim3 block(Configuration::BLOCKX, Configuration::BLOCKY);
		setupRandom << <grid, block >> > (randStates, (unsigned long)time(0));
		Tool::deviceSynchronize();
		package.randStates = randStates;

		//prepare light pointer array
		Light** lights = NULL;
		if (package.numLights > 0)
		{
			Tool::cudaMallocChecked((void**)&lights, package.numLights * sizeof(Light*), "Renderer()::prepareScene()");
			Tool::cudaMemcpyChecked(lights, sceneParser.getLights(), package.numLights * sizeof(Light*),
				cudaMemcpyHostToDevice, "Renderer()::prepareScene()");
		}
		package.lights = lights;

		//prepare object pointer array
		Object3D** objects = NULL;
		if (package.numObjects > 0)
		{
			Tool::cudaMallocChecked((void**)&objects, package.numObjects * sizeof(Object3D*), "Renderer()::prepareScene()");
			Tool::cudaMemcpyChecked(objects, sceneParser.getGroup(), package.numObjects * sizeof(Object3D*),
				cudaMemcpyHostToDevice, "Renderer()::prepareScene()");
		}
		package.objects = objects;

		//prepare light object pointer array
		Object3D** lightobjects = NULL;
		if (package.numLightObjects > 0)
		{
			Tool::cudaMallocChecked((void**)&lightobjects, package.numLightObjects * sizeof(Object3D*), "Renderer()::prepareScene()");
			Tool::cudaMemcpyChecked(lightobjects, sceneParser.getLightObjects(), package.numLightObjects * sizeof(Object3D*),
				cudaMemcpyHostToDevice, "Renderer()::prepareScene()");
		}
		package.lightObjects = lightobjects;
		
		//prepare empty image 
		Vector3f* imageData = NULL;
		Tool::cudaMallocChecked((void**)&imageData, width * height * sizeof(Vector3f), "Render::run()");
		package.image = imageData;

		return package;
	}

	//memory leak will not happen in GPU, but freeing memory is a good habit
	static void freePackage(const Package& package)
	{
		//free package array storage in GPU
		Tool::cudaFreeChecked(package.objects, "Renderer::freePackage() -freeing objects");
		Tool::cudaFreeChecked(package.image, "Renderer::freePackage() -freeing image");
		Tool::cudaFreeChecked(package.lights, "Renderer::freePackage() -freeing lights");
		Tool::cudaFreeChecked(package.lightObjects, "Renderer::freePackage() -freeing light objects");
		Tool::cudaFreeChecked(package.randStates, "Renderer::freePackage() -freeing curandState");
	}

public:

	static void run()
	{
		//timer
		auto start = chrono::high_resolution_clock::now();

		int width, height;
		if (Configuration::SUPERSAMPLING)
		{
			width = Configuration::WIDTH * 3;
			height = Configuration::HEIGHT * 3;
		}
		else
		{
			width = Configuration::WIDTH;
			height = Configuration::HEIGHT;
		}

		//parse scene and prepare a package

		SceneParser sceneParser(Configuration::getInputFile(Configuration::CHOICE));
		if (!sceneParser.checkStatus())
		{
			cout << "Failed to parse scene, error message: " << sceneParser.getErrorMessage() << endl;
			return;
		}

		Package package = prepareScene(width, height, sceneParser);
		Image image(width, height);

		//calculate grid shape according to block shape
		int gridx = (width + Configuration::BLOCKX - 1) / Configuration::BLOCKX;
		int gridy = (height + Configuration::BLOCKY - 1) / Configuration::BLOCKY;
		dim3 grid(gridx, gridy);
		dim3 block(Configuration::BLOCKX, Configuration::BLOCKY);

		//run kernel function and syncronize
		cout << "Start CUDA kernel, grid = (" << gridx << ", " << gridy << "), block = (" << block.x << ", " << block.y << ")" << endl;
		render << <grid, block >> > (package);
		Tool::deviceSynchronize("Renderer::run()");
		cout << "CUDA kernel finished" << endl;

		//transport image fro GPU to CPU and save it
		Tool::cudaMemcpyChecked(image.GetData(), package.image, width * height * sizeof(Vector3f),
			cudaMemcpyDeviceToHost, "Renderer::run()");

		if (Configuration::GAUSSIANBLUR)
		{
			cout << "Start Gaussian blur" << endl;
			image.GaussianBlur();
		}

		if (Configuration::SUPERSAMPLING)
		{
			cout << "Start down sampling" << endl;
			Image small_image(Configuration::WIDTH, Configuration::HEIGHT);
			small_image.DownSampling(image);
			cout << "Save image" << endl;
			small_image.SaveBMP(getOutputFilePath(Configuration::getOutputFile(Configuration::CHOICE)).c_str());
		}
		else
		{
			cout << "Save image" << endl;
			image.SaveBMP(getOutputFilePath(Configuration::getOutputFile(Configuration::CHOICE)).c_str());
		}

		//free storage
		freePackage(package);

		//compute total time
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		int total_seconds = static_cast<int>(diff.count());
		int hours = total_seconds / 3600;
		total_seconds %= 3600;
		int minutes = total_seconds / 60;
		int seconds = total_seconds % 60;
		cout << "Elapsed time: " << hours << ":" << minutes << ":" << seconds << endl;
	}
};