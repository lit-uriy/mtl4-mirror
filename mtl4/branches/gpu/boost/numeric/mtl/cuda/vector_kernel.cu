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

#ifndef MTL_CUDA_VECTOR_KERNEL_INCLUDE
#define MTL_CUDA_VECTOR_KERNEL_INCLUDE

#include <boost/numeric/mtl/cuda/scalar.cu>

namespace mtl { namespace cuda {

template <typename Scalar>
struct vec_rmult_asgn_functor
{
    explicit vec_rmult_asgn_functor(const Scalar& s= 0, Scalar* vec= 0)
      : s(s), vec(vec)  {}

    __device__ __host__ void operator[](int i)
    {	//s.to_device();
        
	  vec[i]*= s;
    }

    Scalar  s;
    Scalar* vec;
};


template <typename Scalar>
struct vec_rmult_asgn
{
    explicit vec_rmult_asgn(const Scalar& s= 0, Scalar* vec= 0, int n= 0)
	: f(s, vec), n(n) 
	{
	  Scalar  *test;
	  cudaMalloc( (void **) &test, sizeof(Scalar));
	  cudaMemcpy(test, &s, sizeof(Scalar), cudaMemcpyHostToDevice); 
	}

    __device__ void operator()(void)
    {
        const unsigned grid_size = blockDim.x * gridDim.x, id= blockIdx.x * blockDim.x + threadIdx.x,
	               blocks= n / grid_size,  nn= blocks * grid_size;

	for (int i = id; i < nn; i+= grid_size)
	    f[i];
	if (nn + id < n)
	    f[nn + id];
    }

    vec_rmult_asgn_functor<Scalar> f;
    int     n;

};

template <typename Scalar>
struct vec_rdivide_asgn_functor
{
    explicit vec_rdivide_asgn_functor(const Scalar& s= 0, Scalar* vec= 0)
      : s(s), vec(vec)  {}

    __device__ __host__ void operator[](int i)
    {	//s.to_device();
        
	  vec[i]/= s;
    }

    Scalar  s;
    Scalar* vec;
};


template <typename Scalar>
struct vec_rdivide_asgn
{
    explicit vec_rdivide_asgn(const Scalar& s= 0, Scalar* vec= 0, int n= 0)
	: f(s, vec), n(n) 
	{
	  Scalar  *test;
	  cudaMalloc( (void **) &test, sizeof(Scalar));
	  cudaMemcpy(test, &s, sizeof(Scalar), cudaMemcpyHostToDevice); 
	}

    __device__ void operator()(void)
    {
        const unsigned grid_size = blockDim.x * gridDim.x, id= blockIdx.x * blockDim.x + threadIdx.x,
	               blocks= n / grid_size,  nn= blocks * grid_size;

	for (int i = id; i < nn; i+= grid_size)
	    f[i];
	if (nn + id < n)
	    f[nn + id];
    }

    vec_rdivide_asgn_functor<Scalar> f;
    int     n;

};


template <typename Scalar>
struct vec_rplus_asgn_functor
{
    explicit vec_rplus_asgn_functor(const Scalar& s= 0, Scalar* vec= 0)
      : s(s), vec(vec)  {}

    __device__ __host__ void operator[](int i)
    {	//s.to_device();
        
	  vec[i]+= s;
    }

    Scalar  s;
    Scalar* vec;
};


template <typename Scalar>
struct vec_rplus_asgn
{
    explicit vec_rplus_asgn(const Scalar& s= 0, Scalar* vec= 0, int n= 0)
	: f(s, vec), n(n) 
	{
	  Scalar  *test;
	  cudaMalloc( (void **) &test, sizeof(Scalar));
	  cudaMemcpy(test, &s, sizeof(Scalar), cudaMemcpyHostToDevice); 
	}

    __device__ void operator()(void)
    {
        const unsigned grid_size = blockDim.x * gridDim.x, id= blockIdx.x * blockDim.x + threadIdx.x,
	               blocks= n / grid_size,  nn= blocks * grid_size;

	for (int i = id; i < nn; i+= grid_size)
	    f[i];
	if (nn + id < n)
	    f[nn + id];
    }

    vec_rplus_asgn_functor<Scalar> f;
    int     n;

};

template <typename Scalar>
struct vec_rminus_asgn_functor
{
    explicit vec_rminus_asgn_functor(const Scalar& s= 0, Scalar* vec= 0)
      : s(s), vec(vec)  {}

    __device__ __host__ void operator[](int i)
    {	//s.to_device();
        
	  vec[i]-= s;
    }

    Scalar  s;
    Scalar* vec;
};


template <typename Scalar>
struct vec_rminus_asgn
{
    explicit vec_rminus_asgn(const Scalar& s= 0, Scalar* vec= 0, int n= 0)
	: f(s, vec), n(n) 
	{
	  Scalar  *test;
	  cudaMalloc( (void **) &test, sizeof(Scalar));
	  cudaMemcpy(test, &s, sizeof(Scalar), cudaMemcpyHostToDevice); 
	}

    __device__ void operator()(void)
    {
        const unsigned grid_size = blockDim.x * gridDim.x, id= blockIdx.x * blockDim.x + threadIdx.x,
	               blocks= n / grid_size,  nn= blocks * grid_size;

	for (int i = id; i < nn; i+= grid_size)
	    f[i];
	if (nn + id < n)
	    f[nn + id];
    }

    vec_rminus_asgn_functor<Scalar> f;
    int     n;

};



template <typename NullaryFunction>
__global__
void launch_function(NullaryFunction f)
{
  f();
}

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_KERNEL_INCLUDE



