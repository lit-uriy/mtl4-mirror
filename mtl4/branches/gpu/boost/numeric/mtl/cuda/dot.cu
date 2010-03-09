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

#include <cassert>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/cuda/dot_kernel.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>

#define BLOCK_SIZE 512

namespace mtl { namespace cuda {


template <typename Vector>
typename mtl::Collection<Vector>::value_type dot( const Vector& v1, const Vector& v2)
{
    assert(size(v1) == size(v2));
    typedef typename mtl::Collection<Vector>::value_type value_type;
    
    dim3 dimGrid( ceil(double(size(v1)) / double(BLOCK_SIZE)) ), dimBlock( BLOCK_SIZE );
    vector<value_type> out(dimBlock.x * dimGrid.x, value_type(0), false);
  //  std::cout<< "dimGrid.x=" << ceil(double(size(v1)) / double(BLOCK_SIZE)) << "\n";
 //   std::cout<< "Produkt=" << dimBlock.x * dimGrid.x << "\n";
    v1.to_device(); v2.to_device();
    dot_kernel<<< dimGrid, dimBlock, dimBlock.x * sizeof(value_type) >>>(out.get_device_pointer(), v1.get_device_pointer(), v2.get_device_pointer(), size(v1));
  //  reduce_kernel_kompliziert<<<dimGrid, dimBlock, dimBlock.x * sizeof(value_type)>>>(out.get_device_pointer(), v1.get_device_pointer(), size(v1), BLOCK_SIZE);
//     value_type temp= 0;
 //   std::cout<< "out=" << out << "\n";
 /*   for (int i= 0; i < dimGrid.x; i++)
      temp+= out[i];
 */   
    return out[0];
}


}} // namespace mtl::cuda
#endif // MTL_CUDA_DOT_INCLUDE

