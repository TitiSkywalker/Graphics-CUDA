/*
* If you want to add some features in this framework:
*   - inherit Object3D/Light/Camera/Material, add a new .cuh file
*   - change the "SceneParser" part to parse this class
*   - change the "Tool"       part to allocate this class in GPU memory
*   - change the "GPURenderer" part to move this class into shared memory in GPU
*   - change the union definition to compute the size of shared memory automatically
*/
#include "CPURenderer.cuh"

int main(int argc, char** argv)
{
    Renderer::run();
}