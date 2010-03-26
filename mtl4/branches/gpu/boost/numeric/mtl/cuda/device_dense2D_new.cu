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


#ifndef MTL_CUDA_DEVICE_DENSE2D_NEW_INCLUDE
#define MTL_CUDA_DEVICE_DENSE2D_NEW_INCLUDE

#include <cstdio>

namespace mtl { namespace cuda {


template <typename T>
T* device_dense2D_new(int num_cols, int num_rows)
{   
    T* pointer;
    cudaMalloc(reinterpret_cast<void **>(&pointer), sizeof(T) * num_cols * num_rows);
    return pointer;
}

template <typename T>
T* device_dense2D_new(const T& value, int num_cols, int num_rows)
{   
    T* pointer= device_dense2D_new<T>(num_cols, num_rows);

    // copy value to first entry and replicate it
    cudaMemcpy(pointer, &value, sizeof(T), cudaMemcpyHostToDevice);\
    int temp= 0;
    for (int i = 0; i < num_rows; i++){
	for (int j = 0; j < num_cols; j++){
	    cudaMemcpy(pointer + temp, pointer, sizeof(T), cudaMemcpyDeviceToDevice);
	    temp++;
	}
    }

    return pointer;
}

}} // namespace mtl::cuda

#endif // MTL_CUDA_DEVICE_DENSE2D_NEW_INCLUDE

