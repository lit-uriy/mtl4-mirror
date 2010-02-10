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
#include <assert.h>
#include <boost/numeric/mtl/cuda/device_vector_new.cu>
//#include <boost/numeric/mtl/cuda/mult_vector_kernel.cu>

namespace mtl { namespace cuda {

/// Class for replicating vectors on host and device
template <typename T>
class vector
{
    typedef vector<T>                self;
  public:
    typedef T                        value_type;
    //static const int dim;
    /// Constructor from type T
 
   vector(int n=1, const T& value= T(), bool on_host=true ) 
  	: dim(n), start(new T[n]), dptr(device_vector_new(value, n)), on_host(on_host) {} 
//  : hvalue(value), dptr(on_host ? device_new<T>()[dim] : device_new(value)), dim(n) , on_host(on_host) {}

    ~vector() {
	 delete [] start; 
	 cudaFree(dptr);
   }

    T& operator[](int index) {
	assert(index >= 0 && index < dim);
	return start[index];
    }

    const T& operator[](int index) const {
        assert(index >= 0 && index < dim);
        return start[index];
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
	to_device();
	return *this;
    }

    // template <typename T>
    self& operator*=(const value_type src)
    {
        std::cout<< "x*=wert zuweisung\n";
   	for (int i= 0; i < dim; i++) {
       		 start[i]*= src;
       	}
       	cudaMemcpy(dptr, start, sizeof(T)*dim, cudaMemcpyHostToDevice);
       	on_host= false;
	dim3 dimGrid(1);
	dim3 dimBlock(dim);
        //mult_vector_kernel<<< (dimGrid, dimBlock)>>>(dptr, src); 
 	cudaMemcpy(start, dptr, sizeof(T)*dim, cudaMemcpyDeviceToHost);
        return *this;
    }
 //    void test (value_type f) {std::cout << "tmp=" << f << "\n"; }
 
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
    }

//     T value() { 
// 	if (!valid_host())
// 	{
// 		cudaMemcpy(start, dptr, sizeof(T)*dim, cudaMemcpyDeviceToHost);
// 	}
// 	return *start;
//     }

//     T const& value() const { const_cast<self*>(this)->to_host(); return start; }

//     operator T&() { return value(); }
//     operator T const&() const { return value(); }
    
    friend std::ostream& operator<<(std::ostream& os, self& x)
    {
	x.replicate_on_host();
	os << "{" << x.size() << (x.valid_host() ? ",host}(" : ",device}(");
	for (int i= 0; i < x.size(); i++)
	    os << x.start[i] << (i < x.dim - 1 ? ", " : ")");

// 	if (x.valid_host()){
// 	    os << "{" << x.size() << ",host}(";
// 	    for (int i= 0; i < x.size()-1; i++){
// 		os << " " << x[i] << ", ";
// 	    }
// 	    os << x[x.size()-1];
// 	    os << ")\n";
//         } else { 
// 	    cudaMemcpy(x.start, x.dptr, sizeof(T)*x.dim, cudaMemcpyDeviceToHost);
// 	    os << "{" << x.size() << ",device}(";
//             for (int i= 0; i < x.size()-1; i++){
//                 os << " " << x[i] << ", ";
//             }
//             os << x[x.size()-1];
//             os << ")\n";

// 	}
	return os;
    }

  protected:
    int  dim;
    T*   start; // Value on host //TODO    malloc sizeof(T)*dim
    T*   dptr;   // Value on device (allocated as pointer whose content is referred)
    bool on_host;
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_INCLUDE
