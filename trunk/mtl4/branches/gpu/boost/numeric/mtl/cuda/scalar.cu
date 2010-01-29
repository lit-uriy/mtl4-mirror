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
    scalar(const T& value= T()) : dptr(device_new(value)) {}

    ~scalar() { cudaFree(dptr); }

    self& operator=(const scalar& that)
    {
	cudaMemcpy(dptr, that.dptr, sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
    }

    self& operator=(const value_type& src)
    {
	cudaMemcpy(dptr, &src, sizeof(T), cudaMemcpyHostToDevice);
	return *this;
    }

    template <typename Src>
    self& operator*=(const Src& src)
    { 
	return *this;
    }	

   
    //  void test (value_type f) {std::cout << "tmp=" << f << "\n"; }
 
    bool valid_host() const { return false; }
    bool valid_device() const { return true; }

    void to_host() {}
    void to_device() {}

    T value() const 
    { 
	T copy; 
	cudaMemcpy(&copy, dptr, sizeof(T), cudaMemcpyDeviceToHost);
	return copy;
    }

//     operator T() const { return value(); }

    friend std::ostream& operator<<(std::ostream& os, const self& x)
    {
	return os << x.value();
    }

  public:
    T* dptr; // Value on device (allocated as pointer whose content is referred)
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_SCALAR_INCLUDE


	/*
// 	if (valid_host())
//         {
// 	    hvalue*= src;
// 	    //   cudaMemcpy(dptr, &hvalue, sizeof(T), cudaMemcpyHostToDevice);
// 	}
// 	else{
// 	    value_type copy;
	    
// 	    cudaMemcpy(&copy, dptr, sizeof(T), cudaMemcpyDeviceToHost);
// 	    std::cout<< "devise_copy_befor=" << copy << "\n";
// 	    copy*= src;;
// 	    std::cout<< "devise_copy_after"<< copy << "\n";
// 	    cudaMemcpy(dptr, &copy, sizeof(T), cudaMemcpyHostToDevice);
// 	}
*/
