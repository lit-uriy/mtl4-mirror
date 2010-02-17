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

#ifndef MTL_CUDA_DOT_INCLUDE
#define MTL_CUDA_DOT_INCLUDE



#include <boost/numeric/mtl/cuda/cuda_dot_kernel.h>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>

namespace mtl { namespace cuda {

//entrance into dot_kernel  (like "main in .cu")
template <typename Vector>
typename mtl::Collection<Vector>::value_type dot(Vector v1, Vector v2)
{
//    typedef typename mtl::Collection<Vector>::value_type value_type; 
	dot(v1,v2),

	std::cout<< "Hallo\n";
       // dot_kernel<<< dimGrid, dimBlock >>>(out, v1.dptr, v2.dptr, v1.dim);

	return 0;
}


}} // namespace mtl::cuda
#endif // MTL_CUDA_DOT_INCLUDE
