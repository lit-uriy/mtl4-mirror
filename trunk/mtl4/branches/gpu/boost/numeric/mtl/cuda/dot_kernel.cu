// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_CUDA_DOT_KERNEL_INCLUDE
#define MTL_CUDA_DOT_KERNEL_INCLUDE

#include <boost/numeric/mtl/cuda/cuda_dot_kernel.h>

namespace mtl { namespace cuda {

template <typename T, unsigned blocksize>
__global__ void reduce_kernel_kompliziert(T* out, T* in, unsigned int n)
{
    extern __shared__ T sdata[];

    //all threads load one element to shared memory
    unsigned int id= threadIdx.x,
	         i = blockIdx.x * blocksize * 2 + id,
	         gridSize = blocksize * 2 * gridDim.x;
    sdata[id]= 0;
    while (i < n){
	sdata[id]+= in[i] + in[i+blocksize];
	i += gridSize;
    }
    __syncthreads();
    
    //reduction in shared memory
    if (blocksize >= 512) {
	if (id < 256) sdata[id]+= sdata[id + 256];
	__syncthreads();
    }
    if (blocksize >= 256) {
	if (id < 128) sdata[id]+= sdata[id + 128];
	__syncthreads();
    }
    if (blocksize >= 128) {
	if (id < 64) sdata[id]+= sdata[id + 64];
	__syncthreads();
    }
    if (id < 32){
	if (blocksize >= 64) sdata[id]+= sdata[id + 32];
	if (blocksize >= 32) sdata[id]+= sdata[id + 16];
	if (blocksize >= 16) sdata[id]+= sdata[id +  8];
	if (blocksize >=  8) sdata[id]+= sdata[id +  4];
	if (blocksize >=  4) sdata[id]+= sdata[id +  2];
	if (blocksize >=  2) sdata[id]+= sdata[id +  1];
    }
    //write result of block to global memory
    if (id == 0) out[blockIdx.x]= sdata[0];

}

template <typename T>
__global__ void reduce_kernel(T* out, T* in, unsigned int n)
{
    extern __shared__ T sdata[];
    unsigned int id= threadIdx.x,i= blockDim.x * gridDim.x + id;

    if (i < n)
	sdata[i]= in[i];
    else
	sdata[i]= 0;

    int j= 1;
    while (j < n) j<<= 1;
    j>>= 1;

    for (; j > 32; j>>= 1) {
	sdata[i]+= sdata[i + j];
	__syncthreads();
    }

    for (; j > 0; j>>= 1)
	sdata[i]+= sdata[i + j];

    //write result of block to global memory
    if (id == 0) out[blockIdx.x]= sdata[0];

}

template <typename T>
__global__ void dot_kernel(T* out, T* v1, T* v2, unsigned n)
{
    extern __shared__ T sdata[];

    //all threads load one element to shared memory
    unsigned int id= threadIdx.x,
	         i = blockIdx.x * gridDim.x + id,
	         step= blockIdx.x * gridDim.x;
    
    sdata[id]= 0;

    //do multiplication parallel on gpu not on cpu
    // divide into blocks of n/step

    unsigned blocks= n / step, nn= blocks * step, rem= n - nn;
    for (int j= id; j < nn; j+= steps)
	sdata[id]+= v1[j] * v2[j];

    if (nn + id < n)
	sdata[id]+= v1[nn + id] * v2[nn + id];

}



}} // namespace mtl::cuda

#endif // MTL_CUDA_DOT_KERNEL_INCLUDE
