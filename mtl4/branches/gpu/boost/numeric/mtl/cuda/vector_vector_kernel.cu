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
__global__ void vector_vector_rminus (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]-=v2[idx];


}


#if 0 // I don't think that we need this

template <typename Vector>
__global__ void vector_vector_rmult (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]*=v2[idx];


}



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

