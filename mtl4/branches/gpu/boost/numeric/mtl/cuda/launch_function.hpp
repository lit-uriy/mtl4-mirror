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

#ifndef MTL_CUDA_LAUNCH_FUNCTION_INCLUDE
#define MTL_CUDA_LAUNCH_FUNCTION_INCLUDE

#ifdef MTL_HAS_CUDA

namespace mtl { namespace cuda {

/// Launch computation given by nullary function (functor) on device
template <typename NullaryFunction>
__global__
void launch_function(NullaryFunction f)
{
  f();
}

}} // namespace mtl::cuda

#endif // MTL_HAS_CUDA

#endif // MTL_CUDA_LAUNCH_FUNCTION_INCLUDE
