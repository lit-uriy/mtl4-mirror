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

#ifndef MTL_CUDA_COMPRESSED2D_KERNEL_INCLUDE
#define MTL_CUDA_COMPRESSED2D_KERNEL_INCLUDE


namespace mtl { namespace cuda {

//------- kernel for sparse Matrix-Vector-operations


template <typename Matrix, typename Vector>
__global__ void sparse_mat_vec_mult( const  unsigned num_rows ,
                                     const  unsigned * ptr ,
                                     const  unsigned * indices ,
                                     const  Matrix    * data ,
                                     const  Vector * x ,
                                            Vector * y )
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    
        if ( row < num_rows ){
                Matrix dot = 0;
                unsigned row_start = ptr [ row ];
                unsigned row_end   = ptr [ row +1];
                for ( unsigned jj = row_start ; jj < row_end ; jj ++)
                        dot += data [ jj ] * x [ indices [ jj ]];
                y [ row ] = dot ;
        }

}

}} // namespace mtl::cuda

#endif // MTL_CUDA_COMPRESSED2D_KERNEL_INCLUDE
