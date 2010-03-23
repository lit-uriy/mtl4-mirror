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

#ifndef MTL_CUDA_VECTOR_VECTOR_KERNEL_INCLUDE
#define MTL_CUDA_VECTOR_VECTOR_KERNEL_INCLUDE

//#include <boost/numeric/mtl/cuda/scalar.cu>

namespace mtl { namespace cuda {

//------- kernel for operation Vector-Vector


template <typename Vector>
__global__ void vector_vector_rplus (Vector *v1, Vector *v2, int dim)
{/*
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]+=v2[idx];*/

unsigned id= blockIdx.x * blockDim.x +  threadIdx.x,
	 left = dim - gridDim.x*blockDim.x,
	 step = left/(gridDim.y),
	 rest = left%(gridDim.y),
	 index = (blockDim.x * gridDim.x -1) + step*(blockIdx.y);


    /// when the vector dimention is smaller than the conbination of blockDim and gridDim
    if(dim<gridDim.x*blockDim.x){	
	/// Stop condition, because, we must stop the calculation, when is equal to  the vector dimention
	if (id<dim) 
	    v1[id]+= v2[id]; 
    } 
    
    
    /// when the vector dimention is bigger than the conbination of blockDim and gridDim
    else{ 
    
    ///  first part of the calculation to achieve the combination of blockDim and gridDim
    if (id < gridDim.x*blockDim.x -1)
       v1[id]+= v2[id];

    /// second we calculate step in each thread, but this part is the step+rest
    if(id == gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step+rest; i++)
	    v1[index+i]+= v2[index+i];   

    /// third we calculate just step in each thread
    if(id > gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step; i++)
	    v1[index+i]+= v2[index+i]; 
	
    } 
 

 
}


///testing without copy
template <typename Vector>
__global__ void vector_vector_assign_plus (Vector *vout, Vector *v1, Vector *v2, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim)
       vout[idx]= v1[idx] + v2[idx];
}




template <typename Vector>
__global__ void vector_vector_rminus (Vector *v1, Vector *v2, int dim)
{/*
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<dim)
       v1[idx]-=v2[idx];*/

unsigned id= blockIdx.x * blockDim.x +  threadIdx.x,
	 left = dim - gridDim.x*blockDim.x,
	 step = left/(gridDim.y),
	 rest = left%(gridDim.y),
	 index = (blockDim.x * gridDim.x -1) + step*(blockIdx.y);


    /// when the vector dimention is smaller than the conbination of blockDim and gridDim
    if(dim<gridDim.x*blockDim.x){	
	/// Stop condition, because, we must stop the calculation, when is equal to  the vector dimention
	if (id<dim) 
	    v1[id]-= v2[id]; 
    } 
    
    
    /// when the vector dimention is bigger than the conbination of blockDim and gridDim
    else{ 
    
    ///  first part of the calculation to achieve the combination of blockDim and gridDim
    if (id < gridDim.x*blockDim.x -1)
       v1[id]-= v2[id];

    /// second we calculate step in each thread, but this part is the step+rest
    if(id == gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step+rest; i++)
	    v1[index+i]-= v2[index+i];   

    /// third we calculate just step in each thread
    if(id > gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step; i++)
	    v1[index+i]-= v2[index+i]; 
	
    } 
 



}


 // I don't think that we need this
template <typename Vector>
__global__ void vector_vector_rmult (Vector *v1, Vector *v2, int dim)
{

/*
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<dim)
        v1[idx]*=v2[idx];
*/

unsigned id= blockIdx.x * blockDim.x +  threadIdx.x,
	 left = dim - gridDim.x*blockDim.x,
	 step = left/(gridDim.y),
	 rest = left%(gridDim.y),
	 index = (blockDim.x * gridDim.x -1) + step*(blockIdx.y);


    /// when the vector dimention is smaller than the conbination of blockDim and gridDim
    if(dim<gridDim.x*blockDim.x){	
	/// Stop condition, because, we must stop the calculation, when is equal to  the vector dimention
	if (id<dim) 
	    v1[id]*= v2[id]; 
    } 
    
    
    /// when the vector dimention is bigger than the conbination of blockDim and gridDim
    else{ 
    
    ///  first part of the calculation to achieve the combination of blockDim and gridDim
    if (id < gridDim.x*blockDim.x -1)
       v1[id]*= v2[id];

    /// second we calculate step in each thread, but this part is the step+rest
    if(id == gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step+rest; i++)
	    v1[index+i]*= v2[index+i];   

    /// third we calculate just step in each thread
    if(id > gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step; i++)
	    v1[index+i]*= v2[index+i]; 
	
    } 
 


}




///kernel plus, for vectors larger than 33 millionen
template <typename T>
__global__ void vector_vector_assign_plus_updated (T *vout, T *v1, T *v2, int dim)
{
unsigned id= blockIdx.x * blockDim.x +  threadIdx.x,
	 left = dim - gridDim.x*blockDim.x,
	 step = left/(gridDim.y),
	 rest = left%(gridDim.y),
	 index = (blockDim.x * gridDim.x -1) + step*(blockIdx.y);


    /// when the vector dimention is smaller than the conbination of blockDim and gridDim
    if(dim<gridDim.x*blockDim.x){	
	/// Stop condition, because, we must stop the calculation, when is equal to  the vector dimention
	if (id<dim) 
	    vout[id]= v1[id] + v2[id]; 
    } 
    
    
    /// when the vector dimention is bigger than the conbination of blockDim and gridDim
    else{ 
    
    ///  first part of the calculation to achieve the combination of blockDim and gridDim
    if (id < gridDim.x*blockDim.x -1)
       vout[id]= v1[id] + v2[id];

    /// second we calculate step in each thread, but this part is the step+rest
    if(id == gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step+rest; i++)
	    vout[index+i]= v1[index+i] + v2[index+i];   

    /// third we calculate just step in each thread
    if(id > gridDim.x*blockDim.x -1)
	for(unsigned i=0; i<=step; i++)
	    vout[index+i]= v1[index+i] + v2[index+i]; 
	
    }
}
///end updated function






#if 0

template <typename Vector>
__global__ void vector_vector_rdivide (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]/=v2[idx];


}

#endif

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_VECTOR_KERNEL_INCLUDE

