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

#ifndef MTL_CUDA_GET_DEVICE_VALUE_INCLUDE
#define MTL_CUDA_GET_DEVICE_VALUE_INCLUDE

namespace mtl { namespace cuda {

/// Get value from constant device pointer \p ptr
template <typename T>
T get_device_value(T* const ptr)
{
    T tmp;
    // Cuda is blind for const-ness (const_cast is save)
    cudaMemcpy(&tmp, const_cast<T*>(ptr), sizeof(T), cudaMemcpyDeviceToHost); 
    return tmp;
}

}} // namespace mtl::cuda

#endif // MTL_CUDA_GET_DEVICE_VALUE_INCLUDE
