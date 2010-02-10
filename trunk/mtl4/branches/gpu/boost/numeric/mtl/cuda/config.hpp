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

#ifndef MTL_CUDA_CONFIG_INCLUDE
#define MTL_CUDA_CONFIG_INCLUDE

namespace mtl { namespace cuda {

#ifdef MTL_CUDA_HOST_LIMIT
    const unsigned host_limit= MTL_CUDA_HOST_LIMIT;
#else
    const unsigned host_limit= 1024;
#endif


}} // namespace mtl::cuda

#endif // MTL_CUDA_CONFIG_INCLUDE
