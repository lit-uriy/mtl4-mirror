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
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]+=v2[idx];


}



template <typename Vector>
__global__ void vector_vector_assign_plus (Vector *vout, Vector *v1, Vector *v2, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim)
       vout[idx]= v1[idx] + v2[idx];
}


///updated function
template <typename T>
__global__ void vector_vector_assign_plus_updated (T *vout, T *v1, T *v2, int dim)
{
    extern __shared__ T sdata[]; //shared memory
/*    
    const unsigned tid= threadIdx.x, 
		   id= blockIdx.x * gridDim.x + tid,
                   step= blockDim.x, 
		   blocks= dim / step, 
		   nn= blocks * step;
    
    T reg(0);

     for (int j= id; j < nn; j+= step)
 	reg+= v1[j] + v2[j];

    if (nn + id < dim)
	reg+= v1[nn + id] + v2[nn + id];
    
    sdata[tid]= reg;
    
    __syncthreads();
      
    if (tid == 0) {
	for (int i= 1; i < blockDim.x; i++)
	   sdata[0]+= sdata[i];
	vout[blockIdx.x]= sdata[0];
    }
    __syncthreads();
	
    vout[5]=blockDim.x;
    vout[6]=gridDim.x;
    vout[7]=step;*/
   
  

///     const unsigned tid= threadIdx.x, 
/// 		   id= blockIdx.x * gridDim.x + tid,
///                   step= blockDim.x, 
/// 		   blocks= dim / step, 
/// 		   nn= blocks * step;



unsigned id= blockIdx.x * blockDim.x +  threadIdx.x;
//unsigned id = (blockIdx.y * 128 * 256) + blockIdx.x * 256 + threadIdx.x;

///unsigned  id_thread= threadIdx.x, id= blockIdx.x * gridDim.x + id_thread;
	  
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;

//if (id > gridDim.x) vout[0]++;

    if (id < dim){
       vout[id]= v1[id] + v2[id];
       
    }
    




if (id == gridDim.x*blockDim.x -1 ) {
    //id= blockIdx.y*gridDim.y + gridDim.x +  threadIdx.y;

    
unsigned index = blockDim.x * gridDim.x + blockDim.x*blockIdx.y;
//vout[dim-blockIdx.y]= index;    
    
    for(unsigned i=index; i<index+blockDim.x*blockIdx.y ; i++)
	vout[i]= v1[i] + v2[i]; 
   
    //vout[dim-blockIdx.y]= blockIdx.x * blockDim.x +  threadIdx.x;//v1[dim-blockIdx.y] + v2[dim-blockIdx.y];   
   }






}
///end updated function











template <typename Vector>
__global__ void vector_vector_rminus (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]-=v2[idx];


}


 // I don't think that we need this

template <typename Vector>
__global__ void vector_vector_rmult (Vector *v1, Vector *v2, int dim)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx<dim)
       v1[idx]*=v2[idx];


}

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

