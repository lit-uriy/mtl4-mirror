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

//testing...

#include </usr/local/cuda/include/__cudaFatFormat.h>
#include </usr/local/cuda/include/host_defines.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/cufft.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/driver_types.h>
#include </usr/local/cuda/include/common_types.h>
#include </usr/local/cuda/include/device_types.h>
#include </usr/local/cuda/include/host_config.h>
#include <iostream>
#include <boost/numeric/mtl/cuda/device_new.hpp>

namespace mtl { namespace cuda {


template<class T>
class scalar {

typedef cuda::scalar< T> self;

public: typedef T value_type;



scalar(const T &value = T()) : dptr((device_new(value))) { }

~scalar() { cudaFree(this->dptr); }

self &operator=(const cuda::scalar< T> &that)
{
cudaMemcpy(this->dptr, that.dptr, sizeof(T), cudaMemcpyDeviceToDevice);
return *this;
}

self &operator=(const value_type &src)
{
cudaMemcpy(this->dptr, &src, sizeof(T), cudaMemcpyHostToDevice);
return *this;
}

template<class Src> self &
operator*=(const Src &src)
{
return *this;
}




bool valid_host() const { return false; }
bool valid_device() const { return true; }

void to_host() { }
void to_device() { }

T value() const
{
auto T copy;
cudaMemcpy(&copy, this->dptr, sizeof(T), cudaMemcpyDeviceToHost);
return copy;
}



friend inline std::ostream &operator<<(std::ostream &os, const self &x)
{
return os << x.value();
}


T *dptr;
};

}}


#endif // MTL_CUDA_SCALAR_INCLUDE


