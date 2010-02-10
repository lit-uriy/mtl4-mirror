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
    explicit vec_rscale_asgn(const Scalar& s, Scalar* vec) 
      : s(s), vec(vec) {}

    __host__ __device__ void operator()()
    {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	vec[idx]*= s;
    }

    Scalar s;
    Scalar* vec;
};
 

template<typename NullaryFunction>
__global__
void launch_function(NullaryFunction f)
{
  f();
}


}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_KERNEL_INCLUDE
