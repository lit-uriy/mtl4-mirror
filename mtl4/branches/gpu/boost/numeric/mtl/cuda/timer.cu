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

#include <sys/time.h>

namespace mtl { namespace cuda {

/// Measure elapsed time
class timer
{
  public:
    /// Default constructor
    timer() { restart(); }
    
    /// Restart
    void restart() {  gettimeofday(&start, NULL); }
    
    /// Elapsed time in seconds
    double elapsed() const
    {
        timeval stop;
	gettimeofday(&stop, NULL);
	double t1= start.tv_sec + start.tv_usec / 1000000.0, 
	       t2= stop.tv_sec + stop.tv_usec / 1000000.0;
	return t2 - t1;
    }
  private:
     timeval start;
};


}} // namespace mtl::cuda

#endif // MTL_CUDA_TIMER_CU_INCLUDE

