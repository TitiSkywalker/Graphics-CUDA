//This is the configuration file that sets different parameters.
#pragma once

//the useful "epsilon" everywhere
static constexpr float EPSILON = 0.01f;

//color falloff for secondary rays
static constexpr float FALLOFF = 0.025f;

class Configuration
{
public:
	//image size
	static const int WIDTH = 512;
	static const int HEIGHT = 512;

	//the shape of each thread block(warning: too many threads will cause stack overflow)
	static const int BLOCKX = 32;
	static const int BLOCKY = 16;
	
	//ray tracing parameter
	static const int SAMPLERATE = 20;
	static const int MAXDEPTH = 10;

	//perform anti-aliasing?
	static const bool SUPERSAMPLING = true;	
	static const bool GAUSSIANBLUR = true;		
	
	//input and output files
	static const int CHOICE = 0;
	static const char* getInputFile(int choice)
	{
		switch (choice)
		{
		case 0:
			return "scene0_sphere.scene";
		default:
			return "scene0_sphere.scene";
		}
	}

	static const char* getOutputFile(int choice)
	{
		switch (choice)
		{
		case 0:
			return "scene0_sphere.bmp";
		default:
			return "scene0_sphere.bmp";
		}
	}
};