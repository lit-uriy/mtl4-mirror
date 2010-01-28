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

// Here come Cuda includes ...

//testing...

#include </usr/local/cuda/include/__cudaFatFormat.h>
#include </usr/local/cuda/include/host_defines.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/cufft.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/driver_types.h>
#include </usr/local/cuda/include/common_types.h>
#include </usr/local/cuda/include/device_types.h>
#include </usr/local/cuda/include/host_config.h>
#include <stdio.h>

namespace mtl { namespace cuda {


template<class T> T *
device_new()
{
auto T *pointer;
cudaMalloc(reinterpret_cast< void **>(&pointer), sizeof(T));
return pointer;
}

template<class T> T *
device_new(const T &value)
{
auto T *pointer = (device_new< T> ());
cudaMemcpy(pointer, &value, sizeof(T), cudaMemcpyHostToDevice);
return pointer;
}


}}


#endif // MTL_CUDA_DEVICE_NEW_INCLUDE


