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

#ifndef MTL_CHECK_INCLUDE
#define MTL_CHECK_INCLUDE

#include <boost/numeric/mtl/config.hpp>

namespace mtl {

#ifdef MTL_HAS_CUDA
    // __host__ void check(bool exp) { assert(exp); }
    __device__ __host__ void check(bool) { }
#else
    void inline check(bool exp) { assert(exp); }
#endif


} // namespace mtl

#endif // MTL_CHECK_INCLUDE
