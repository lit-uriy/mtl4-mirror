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

#ifndef MTL_CUDA_SCALAR_INCLUDE
#define MTL_CUDA_SCALAR_INCLUDE

// Here come Cuda includes ...

#include <cuda/cuda.h>

#include <boost/numeric/mtl/cuda/device_new.hpp>

namespace mtl { namespace cuda {


template<class T>
class scalar {


public: typedef T value_type;



scalar(const T &value = T()) : hvalue(value), dvalue((*(device_new(value)))), on_host(true)
{ }

~scalar() { cudaFree(&(this->dvalue)); }

bool valid_host() const { return this->on_host; }
bool valid_device() const { return !(this->on_host); }

void to_host()
{
if (!(this->on_host)) {
cudaMemcpy(&(this->hvalue), &(this->dvalue), sizeof(T), cudaMemcpyDeviceToHost);
(this->on_host) = true;
}
}

void to_device()
{
if (this->on_host) {
cudaMemcpy(&(this->dvalue), &(this->hvalue), sizeof(T), cudaMemcpyHostToDevice);
(this->on_host) = false;
}
}

operator T() { this->to_host(); return this->hvalue; }


private: T hvalue;
T &dvalue;
bool on_host;
};

}}


#endif // MTL_CUDA_SCALAR_INCLUDE


