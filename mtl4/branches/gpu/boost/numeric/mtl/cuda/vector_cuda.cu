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

#ifndef MTL_CUDA_VECTOR_INCLUDE
#define MTL_CUDA_VECTOR_INCLUDE
//for testing only



#include <iostream>
#include <boost/numeric/mtl/cuda/device_new.cu>

namespace mtl { namespace cuda {

/// Class for replicating scalars on host and device
template <typename T>
class vector
{
    typedef vector<T>                self;
  public:
    typedef T                        value_type;
    static const int dim;
    /// Constructor from type T
    /** Be aware that the constructor is implicit. **/
    vector(const T& value= T(), int n=1, bool on_host=true ) 
      : hvalue(value), dptr(on_host ? device_new<T>()[dim] : device_new(value)), dim(n) , on_host(on_host) {}

    ~vector() { cudaFree(dptr); }

    self& operator=(const vector& that)
    {
	on_host= that.on_host;
	if (on_host){
	    std::cout<< "vector on HOST TEST\n" ;
	    hvalue= that.hvalue;
	}
	else{
	    std::cout<< "vector on Device TEST\n" ;
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

    template <typename T>
    T& operator()(const unsigned int dimension)
    {
	MTL_THROW_IF((dimension < 0 || dimension < dim ,incompatible_size()); 
	
	if (on_host)
	{
	    single_hvalue= hvalue[dimension] ;
	}
	else{
	    std::cout<< "Device TEST\n" ;
	    T copy;
	    cudaMemcpy(&copy, dptr, sizeof(T), cudaMemcpyDeviceToHost);
	    single_hvalue= copy[dimension];
	}
	return single_hvalue;
    }
//    void test (value_type f) {std::cout << "tmp=" << f << "\n"; }
 
    bool valid_host() const { return on_host; }
    bool valid_device() const { return !on_host; }
    int  size() const { return size_of; }

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

    private:
    int dim;
    T  *hvalue; // Value on host //TODO    malloc sizeof(T)*dim
    T* dptr;   // Value on device (allocated as pointer whose content is referred)
    bool on_host;
    T   single_hvalue;
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_INCLUDE
