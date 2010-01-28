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

#ifndef MTL_CUDA_DEVICE_NEW_INCLUDE
#define MTL_CUDA_DEVICE_NEW_INCLUDE

#include <stdio.h>

namespace mtl { namespace cuda { 


template <typename T>
T* device_new()
{
    T* pointer;
    cudaMalloc(reinterpret_cast<void **>(&pointer), sizeof(T));
    return pointer;
}

template <typename T>
T* device_new(const T& value)
{
    T* pointer= device_new<T>();
    cudaMemcpy(pointer, &value, sizeof(T), cudaMemcpyHostToDevice);
    return pointer;
}


}} // namespace mtl::cuda

#endif // MTL_CUDA_DEVICE_NEW_INCLUDE
