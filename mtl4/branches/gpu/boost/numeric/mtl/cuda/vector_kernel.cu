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

#ifndef MTL_CUDA_VECTOR_KERNEL_INCLUDE
#define MTL_CUDA_VECTOR_KERNEL_INCLUDE

namespace mtl { namespace cuda {

template <typename Scalar>
struct vec_rscale_asgn
{
    explicit vec_rscale_asgn(const Scalar& s= 0, Scalar* vec= 0, int n= 0)
      : s(s), vec(vec), n(n) {}

    __device__ void operator()(void)
    {
        const unsigned grid_size = blockDim.x * gridDim.x, id= blockIdx.x * blockDim.x + threadIdx.x,
	               blocks= n / grid_size,  nn= blocks * grid_size;

	for (int i = id; i < nn; i+= grid_size)
	    vec[i]*= s;
	if (nn + id < n)
	    vec[nn + id]*= s;
    }

    Scalar s;
    Scalar* vec;
    int     n;
};


template <typename NullaryFunction>
__global__
void launch_function(NullaryFunction f)
{
  f();
}

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_KERNEL_INCLUDE



