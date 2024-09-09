//This is the configuration file that sets different parameters.
#pragma once

//the useful "epsilon" everywhere
static constexpr float EPSILON = 0.01f;

class Configuration
{
public:
	//image size
	static const int WIDTH = 128;
	static const int HEIGHT = 128;

	//the shape of each thread block(warning: too many threads will make stack overflow)
	static const int BLOCKX = 32;
	static const int BLOCKY = 16;
	
	//ray tracing parameter
	static const int SAMPLERATE = 10;
	static const int MAXDEPTH = 10;

	//perform anti-aliasing?
	static const bool SUPERSAMPLING = false;	//this will make program 9x slower
	static const bool GAUSSIANBLUR = false;		//this is relatively fast
	
	//input and output files
	static const int CHOICE = 1;
	static const char* getInputFile(int choice)
	{
		switch (choice)
		{
		case 1:
			return "scene0_sphere.scene";
		default:
			return "scene0_sphere.scene";
		}
	}

	static const char* getOutputFile(int choice)
	{
		switch (choice)
		{
		case 1:
			return "scene0_sphere.bmp";
		default:
			return "scene0_sphere.bmp";
		}
	}
};