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

#include <iostream>

#include <boost/numeric/mtl/cuda/vector_kernel.cu>
#include <boost/numeric/mtl/cuda/device_new.cu>
#include <boost/numeric/mtl/cuda/device_vector_new.cu>

void print(int arr[])
{
    for (int i= 0; i < 5; i++)
	std::cout << arr[i] << ",";
    std::cout << '\n';
}

template <typename T>
struct scale_functor
{
    explicit scale_functor(T s, T* p) : s(s), p(p) {}

    __device__ void operator()()
    {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	p[idx]*= s;
    }
    
    T  s;
    T* p;
};

template<typename NullaryFunction>
__global__
void launch_function(NullaryFunction f)
{
  f();
}

__global__
void dings_bums(int* p)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    p[idx]= 10-idx;
}

int main(int argc, char* argv[])
{
    int arr[]= {1, 2, 3, 4, 5};

    print(arr);
    
    int* dptr= mtl::cuda::device_vector_new<int>(5);
    cudaMemcpy(dptr, arr, sizeof(arr), cudaMemcpyHostToDevice);

    dim3 dimGrid(1), dimBlock(5); 
    dings_bums<<<dimGrid, dimBlock>>>(dptr);

    scale_functor<int> sc(3, dptr);
    launch_function<<<dimGrid, dimBlock>>>(sc);

    cudaMemcpy(arr, dptr, sizeof(arr), cudaMemcpyDeviceToHost);
    print(arr);

    return 0;
}

#if 0




#endif 
