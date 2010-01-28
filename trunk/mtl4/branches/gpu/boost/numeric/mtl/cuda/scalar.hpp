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



scalar(const T &value = T(), bool on_host = true) : hvalue(value), dptr((on_host ? (device_new< T> ()) : (device_new(value)))), on_host(on_host)
{ }

~scalar() { cudaFree(this->dptr); }

self &operator=(const cuda::scalar< T> &that)
{
(this->on_host) = that.on_host;
if (this->on_host) {
(std::cout << ("Scalar HOST TEST\n"));
(this->hvalue) = (that.hvalue);
} else
{
(std::cout << ("Scalar Device TEST\n"));
cudaMemcpy(this->dptr, that.dptr, sizeof(T), cudaMemcpyDeviceToDevice);
}
return *this;
}

self &operator=(const value_type &src)
{
if (this->on_host) {
(std::cout << ("HOST TEST\n"));
(this->hvalue) = src;
} else
{
(std::cout << ("Device TEST\n"));
cudaMemcpy(this->dptr, &src, sizeof(T), cudaMemcpyHostToDevice);
}
return *this;
}


self &operator*=(value_type &src)
{

if (this->valid_host())
{
(this->hvalue) = (this->hvalue) * src;

} else
{
auto value_type tmp = 0;
cudaMemcpy(&(this->hvalue), this->dptr, sizeof(T), cudaMemcpyDeviceToHost);
tmp = 0;
cudaMemcpy(this->dptr, &tmp, sizeof(T), cudaMemcpyHostToDevice);
}

return *this;
}




bool valid_host() const { return this->on_host; }
bool valid_device() const { return !(this->on_host); }

void to_host()
{
if (!(this->on_host)) {
cudaMemcpy(&(this->hvalue), this->dptr, sizeof(T), cudaMemcpyDeviceToHost);
(this->on_host) = true;
}
}

void to_device()
{
if (this->on_host) {
cudaMemcpy(this->dptr, &(this->hvalue), sizeof(T), cudaMemcpyHostToDevice);
(this->on_host) = false;
}
}

T &value() {
if (!this->valid_host())
{
cudaMemcpy(&(this->hvalue), this->dptr, sizeof(T), cudaMemcpyDeviceToHost);
}
return this->hvalue;
}

const T &value() const { (*(const_cast< self *>(this))).to_host(); return this->hvalue; }

operator T &() { return this->value(); }
operator const T &() const { return this->value(); }

friend inline std::ostream &operator<<(std::ostream &os, const self &x)
{
if (x.on_host) {
os << (x.hvalue); } else
{
auto T copy;
cudaMemcpy(&copy, x.dptr, sizeof(T), cudaMemcpyDeviceToHost);
os << copy;
}
return os;
}


T hvalue;
T *dptr;
bool on_host;
};

}}


#endif // MTL_CUDA_SCALAR_INCLUDE


