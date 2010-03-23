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

#ifndef MTL_CUDA_VECTOR_VECTOR_KERNEL_INCLUDE
#define MTL_CUDA_VECTOR_VECTOR_KERNEL_INCLUDE

//#include <boost/numeric/mtl/cuda/scalar.cu>

namespace mtl { namespace cuda {

//------- kernel for operation Vector-Vector


template <typename Vector>
__global__ void vector_vector_rplus (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]+=v2[idx];


}



template <typename Vector>
__global__ void vector_vector_assign_plus (Vector *vout, Vector *v1, Vector *v2, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim)
       vout[idx]= v1[idx] + v2[idx];
}





///kernel plus, for vectors larger than 33 millionen
template <typename T>
__global__ void vector_vector_assign_plus_updated (T *vout, T *v1, T *v2, int dim)
{
unsigned id= blockIdx.x * blockDim.x +  threadIdx.x,
	 left = dim - gridDim.x*blockDim.x,
	 step = left/(gridDim.y),
	 rest = left%(gridDim.y),
	 index = (blockDim.x * gridDim.x -1) + step*(blockIdx.y);


    if (id < gridDim.x*blockDim.x -1)
       vout[id]= v1[id] + v2[id];


    if(id == gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step+rest; i++)
	    vout[index+i]= v1[index+i] + v2[index+i];   
	
	
    if(id > gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step; i++)
	    vout[index+i]= v1[index+i] + v2[index+i]; 
	
	//vout[dim-5]=dim-index;
// 	vout[dim-4]=rest;
// 	vout[dim-3]=left;
// 	vout[dim-2]=step;
// 	vout[dim-1]=index;
	



}
///end updated function











template <typename Vector>
__global__ void vector_vector_rminus (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]-=v2[idx];


}


 // I don't think that we need this

template <typename Vector>
__global__ void vector_vector_rmult (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]*=v2[idx];


}

#if 0

template <typename Vector>
__global__ void vector_vector_rdivide (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]/=v2[idx];


}

#endif

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_VECTOR_KERNEL_INCLUDE

