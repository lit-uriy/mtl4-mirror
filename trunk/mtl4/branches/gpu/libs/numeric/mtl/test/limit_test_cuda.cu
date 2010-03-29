#include <iostream>
#include <assert.h>
#define BLOCK_SIZE 512
__global__ void vector_plus(int *d_out, int *d_in, int dim)
{
    unsigned int block_id=  blockIdx.x + gridDim.x * blockIdx.y;
    unsigned int thread_id= blockDim.x * block_id + threadIdx.x;
    if(thread_id < dim)
        d_out[thread_id] += d_in[thread_id];
}

////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    // pointer for host memory and size
    int *h_a;
    int dimA = 500*1000*1000;

    int *d_b, *d_a;

    size_t memSize = dimA * sizeof(int);
    h_a = (int *) malloc(memSize);

    cudaMalloc( (void **) &d_a, memSize );

    cudaMalloc( (void **) &d_b, memSize );

    for (int i = 0; i < dimA; i++)
        h_a[i] = 2;
    cudaMemcpy( d_a, h_a, memSize, cudaMemcpyHostToDevice );
    for (int i = 0; i < dimA; i++)
        h_a[i] = 1;
    cudaMemcpy( d_b, h_a, memSize, cudaMemcpyHostToDevice );
    dim3 dimBlock(BLOCK_SIZE);
    int num_blocks = (int) ((float) (dimA + BLOCK_SIZE - 1) / (float) BLOCK_SIZE);
    int max_blocks_per_dimension = 65535;
    int num_blocks_y = (int) ((float) (num_blocks + max_blocks_per_dimension - 1) / (float) max_blocks_per_dimension);
    int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
    dim3 grid_size(num_blocks_x, num_blocks_y, 1);
    std::cout<< "dimA/65535/65535= "<< double((dimA/65535)/double(65535)) << "\n";
    std::cout<< "dimA/65535= "<< dimA/65535 << "\n";
    std::cout<< "num_blocks= "<< num_blocks << "\n";
    std::cout<< "num_blocks_x= "<< num_blocks_x << "\n";
    std::cout<< "num_blocks_y= "<< num_blocks_y << "\n";

    vector_plus<<< grid_size, dimBlock >>>( d_b, d_a , dimA);
    cudaThreadSynchronize();
    cudaMemcpy( h_a, d_b, memSize, cudaMemcpyDeviceToHost );
    for (int i = 0; i < 20; i++)
      assert(h_a[i] == 3 );
      std::cout<< "h_a[" << dimA-1 << "]=" << h_a[dimA-1] << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    return 0;
}

