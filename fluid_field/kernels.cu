#include <pycuda-helpers.hpp>
#define CHUNK_WIDTH

texture<float, 2> fieldTex;

__global__ void diffuse(const int input_width, const int input_height, float *field_in, float *field_out)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


}