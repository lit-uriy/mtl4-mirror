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
#include <cassert>

#include <boost/numeric/mtl/cuda/config.hpp>
#include <boost/numeric/mtl/cuda/get_device_value.cu>
#include <boost/numeric/mtl/cuda/device_vector_new.cu>
#include <boost/numeric/mtl/cuda/vector_kernel.cu>

namespace mtl { namespace cuda {

/// Class for replicating vectors on host and device
template <typename T>
class vector
{
    typedef vector<T>                self;
  public:
    typedef T                        value_type;

    /// Constructor from type T 
    vector(int n=1, const T& value= T(), bool on_host=true ) 
      : dim(n), start(new T[n]), dptr(device_vector_new<T>(n)), on_host(on_host) 
    { *this= value; } 

    ~vector() {
	 delete [] start; 
	 cudaFree(dptr);
    }


    //template<typename T>
    vector<T>(const vector<T> &that){   //that Konstruktor
	dim= that.dim;
	start= new T[dim];
	on_host= that.on_host;
	if (on_host)
	    for (int i= 0; i < dim; i++)
		start[i]= that.start[i];
	else
	    cudaMemcpy(dptr, that.dptr, dim*sizeof(T), cudaMemcpyDeviceToDevice);
    }

    //template<typename T>
    self& operator=(const self& that)
    {
	std::cout<< "x= y zuweisung\n";
	std::cout<< "that.on_host="<< that.on_host << "\n";
	assert(dim == that.dim);

	if (this != &that) {  //unnoetige Zuweisung vermeiden
	    on_host= that.on_host;
	    if (on_host) {
		for (int i= 0; i < dim; i++)
		    start[i]= that.start[i];
	    } else
		cudaMemcpy(dptr, that.dptr, dim*sizeof(T), cudaMemcpyDeviceToDevice);
	}
	return *this;
    }

    // Expensive !!!
    template <typename U>
    self& operator=(const vector<U>& that)
    {
	that.replicate_on_host();
	on_host= true;
	for (int i= 0; i < dim; i++)
	    start[i]= that.start[i];
	return *this;
    }


    template <typename U>
    self& operator=(const U& src)
    {	
	std::cout<< "x=wert zuweisung\n";
        for (int i= 0; i < dim; i++) 
            start[i]= src;
	if (!on_host) { on_host= true; to_device(); }
	return *this;
    }

    template <typename U>
    self& operator*=(const U& src)
    {
        std::cout<< "x*= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    std::cout<< "on host\n";
	    for (int i= 0; i < dim; i++) 
		start[i]*= src;
	} else {
	    std::cout<< "on device\n";
	    to_device(); // if not yet there
	    dim3 dimGrid(1), dimBlock(dim); 
	    vec_rscale_asgn<value_type> sc(src, dptr);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }

    T& operator[](int index) {
	assert(index >= 0 && index < dim);
	to_host();
	return start[index];
    }

    T operator[](int i) const 
    {
        assert(i >= 0 && i < dim);
	return on_host ? start[i] : get_device_value(dptr + i);
    }

    bool valid_host() const { return on_host; }
    bool valid_device() const { return !on_host; }
    int  size() const { return dim; }

    void to_host()
    {
	if (!on_host) {
	    cudaMemcpy(start, dptr, sizeof(T)*dim, cudaMemcpyDeviceToHost);
	    on_host= true;
	}
    }

    void replicate_on_host() const
    {
	if (!on_host) 
	    cudaMemcpy(const_cast<self*>(this)->start, dptr, sizeof(T)*dim, cudaMemcpyDeviceToHost);
    }

    void to_device()
    {
	if (on_host) {
	    cudaMemcpy(dptr, start, sizeof(T)*dim, cudaMemcpyHostToDevice);
	    on_host= false;
	}
	for (int i= 0; i < dim; i++) 
            start[i]= 77;
    }
    
    friend std::ostream& operator<<(std::ostream& os, self& x)
    {
	x.replicate_on_host();
	os << "{" << x.size() << (x.valid_host() ? ",host}(" : ",device}(");
	for (int i= 0; i < x.size(); i++)
	    os << x.start[i] << (i < x.dim - 1 ? ", " : ")");
	return os;
    }

  
    int  dim;
    T*   start; // Value on host //TODO    malloc sizeof(T)*dim
    T*   dptr;   // Value on device (allocated as pointer whose content is referred)
    bool on_host;
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_INCLUDE
