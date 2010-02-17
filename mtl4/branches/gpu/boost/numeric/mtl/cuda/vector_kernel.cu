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
        const int grid_size = blockDim.x * gridDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < n) {
	    vec[i]*= s;
	    i+= grid_size;
	}
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



