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

#include <boost/numeric/mtl/cuda/device_new.cu>

namespace mtl { namespace cuda {

/// Class for replicating scalars on host and device
template <typename T>
class scalar
{
    typedef scalar<T>                self;
  public:
    typedef T                        value_type;

    /// Constructor from type T
    /** Be aware that the constructor is implicit. **/
    scalar(const T& value= T()) 
      : hvalue(value), dptr(device_new(value)), on_host(true) {}

    ~scalar() { cudaFree(&dvalue); }

    self& operator=(const scalar& that)
    {
	on_host= that.on_host;
	if (on_host)
	    hvalue= that.hvalue;
	else
	    cudaMemcpy(&hvalue, &dvalue, sizeof(T), cudaMemcpyDeviceToHost);  //richtige Richtung??
	return *this;
    }

    self& operator=(const value_type& src)
    {
	on_host= true;
	hvalue= src;
	return *this;
    }
	
    bool valid_host() const { return on_host; }
    bool valid_device() const { return !on_host; }

    void to_host()
    {
	if (!on_host) {
	    cudaMemcpy(&hvalue, &dvalue, sizeof(T), cudaMemcpyDeviceToHost);
	    on_host= true;
	}
    }

    void to_device()
    {
	if (on_host) {
	    cudaMemcpy(&dvalue, &hvalue, sizeof(T), cudaMemcpyHostToDevice);
	    on_host= false;
	}
    }

    operator T&() { to_host(); return hvalue; }
    operator T const&() const { const_cast<self*>(this)->to_host(); return hvalue; }

  private:
    T  hvalue; // Value on host
    T* dptr; // Value on device (allocated as pointer whose content is referred)
    bool on_host;
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_SCALAR_INCLUDE
