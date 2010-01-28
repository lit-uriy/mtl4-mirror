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

#include <iostream>
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
    scalar(const T& value= T(), bool on_host= true) 
      : hvalue(value), dptr(on_host ? device_new<T>() : device_new(value)), on_host(on_host) {}

    ~scalar() { cudaFree(dptr); }

    self& operator=(const scalar& that)
    {
	on_host= that.on_host;
	if (on_host){
	    std::cout<< "Scalar HOST TEST\n" ;
	    hvalue= that.hvalue;
	}
	else{
	    std::cout<< "Scalar Device TEST\n" ;
	    cudaMemcpy(dptr, that.dptr, sizeof(T), cudaMemcpyDeviceToDevice);
	}
	return *this;
    }

    self& operator=(const value_type& src)
    {
	if (on_host){
	    std::cout<< "HOST TEST\n" ;
	    hvalue= src;
	}
	else{
	    std::cout<< "Device TEST\n" ;
	    cudaMemcpy(dptr, &src, sizeof(T), cudaMemcpyHostToDevice);
	}
	return *this;
    }

    
    self& operator*=( value_type& src)
    { 
	
	if (valid_host())
        {
               hvalue= hvalue * src;
	       //   cudaMemcpy(dptr, &hvalue, sizeof(T), cudaMemcpyHostToDevice);
	}
	else{
	    value_type tmp=0;
	    cudaMemcpy(&hvalue, dptr, sizeof(T), cudaMemcpyDeviceToHost);
	    tmp=0;
            cudaMemcpy(dptr, &tmp, sizeof(T), cudaMemcpyHostToDevice);
	}

	return *this;
    }	

   
    //  void test (value_type f) {std::cout << "tmp=" << f << "\n"; }
 
    bool valid_host() const { return on_host; }
    bool valid_device() const { return !on_host; }

    void to_host()
    {
	if (!on_host) {
	    cudaMemcpy(&hvalue, dptr, sizeof(T), cudaMemcpyDeviceToHost);
	    on_host= true;
	}
    }

    void to_device()
    {
	if (on_host) {
	    cudaMemcpy(dptr, &hvalue, sizeof(T), cudaMemcpyHostToDevice);
	    on_host= false;
	}
    }

    T& value() { 
	if (!valid_host())
	{
		cudaMemcpy(&hvalue, dptr, sizeof(T), cudaMemcpyDeviceToHost);
	}
	return hvalue;
    }

    T const& value() const { const_cast<self*>(this)->to_host(); return hvalue; }

    operator T&() { return value(); }
    operator T const&() const { return value(); }
    
    friend std::ostream& operator<<(std::ostream& os, const self& x)
    {
	if (x.on_host)
	    os << x.hvalue;
	else {
	    T copy; 
	    cudaMemcpy(&copy, x.dptr, sizeof(T), cudaMemcpyDeviceToHost);
	    os << copy;
	}
	return os;
    }

    public:
    T  hvalue; // Value on host
    T* dptr; // Value on device (allocated as pointer whose content is referred)
    bool on_host;
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_SCALAR_INCLUDE
