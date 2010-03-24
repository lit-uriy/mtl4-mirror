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

#ifndef MTL_CUDA_DENSE2D_KERNEL_INCLUDE
#define MTL_CUDA_DENSE2D_KERNEL_INCLUDE


namespace mtl { namespace cuda {

//------- kernel for Matrix-Vector-operations


template <typename Matrix, typename Vector>
__global__ void mat_vec_mult (Vector *b, Matrix *A, Vector *x, int num_rows, int num_cols)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 if (idx < num_rows){
	b[idx]= 0;
 	for(int i= 0; i < num_cols; i++){
 	    b[idx]+= A[i + idx * num_rows] * x[i];
 	}
 }

}


template <typename Matrix, typename T>
__global__ void laplacian (Matrix *A, T d, int num_rows)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x,
     index=idx;

 if (idx < num_rows){
	
        if(idx==0){ 
	    A[0]=d;
	    A[1]=-1;
	}
     
        if(idx>0 && idx<num_rows-1){ 
	    A[index + index*num_rows-1]=-1;
	    A[index + index*num_rows]=d;
	    A[index + index*num_rows+1]=-1;
	}
     
        if(idx==num_rows -1){ 
	    A[index + index*num_rows-1]=-1;
	    A[index + index*num_rows]=d;
	}
     
 	
 }

}



}} // namespace mtl::cuda

#endif // MTL_CUDA_DENSE2D_KERNEL_INCLUDE

