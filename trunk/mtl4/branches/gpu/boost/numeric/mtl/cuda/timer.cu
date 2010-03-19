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

#ifndef MTL_CUDA_TIMER_CU_INCLUDE
#define MTL_CUDA_TIMER_CU_INCLUDE

namespace mtl { namespace cuda {

class timer
{
  public:
    timer()
    {
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );	
    }
    
    float elapsed() const
    {
	cudaEventRecord( stop, 0 );
	float elapsedTimeInMs= 0.0f;
	cudaEventElapsedTime( &elapsedTimeInMs, start, stop );
	return elapsedTimeInMs;
    }
   
    void restart()
    {
	cudaEventRecord( start, 0 );		
    }
  private:
    mutable cudaEvent_t start, stop;
};


}} // namespace mtl::cuda

#endif // MTL_CUDA_TIMER_CU_INCLUDE
