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

#include <boost/numeric/mtl/cuda/config.cu>
#include <boost/numeric/mtl/cuda/device_vector_new.cu>
#include <boost/numeric/mtl/cuda/get_device_value.cu>
#include <boost/numeric/mtl/cuda/meet_data.cu>
#include <boost/numeric/mtl/cuda/vector_kernel.cu>
#include <boost/numeric/mtl/cuda/vector_vector_kernel.cu>

#include <boost/numeric/mtl/vector/vec_expr.hpp>
#include <boost/numeric/mtl/vector/crtp_base_vector.hpp>


//#include </usr/local/cuda/include/cuda_runtime_api.h>

#define BLOCK_SIZE 512


namespace mtl { namespace cuda {

/// Class for replicating vectors on host and device
template <typename T>
class vector
  : public mtl::vector::vec_expr<vector<T> >,
    public mtl::vector::crtp_base_vector< vector<T>, T, std::size_t >
{
    typedef vector<T>                self;
  public:
    typedef T                        value_type;
    typedef std::size_t              size_type;
    typedef mtl::vector::crtp_vector_assign< self, T, size_type >  assign_base;

    /// Constructor from type T 
    vector(int n=1, const T& value= T(), bool on_host=false ) 
      : dim(n), start(new T[n]), dptr(device_vector_new<T>(n)), on_host(on_host) 
    { *this= value; } 

    ~vector() {
	 delete [] start; 
	 cudaFree(dptr);
    }

    void delay_assign() const {} // dummy

//Vector-Vector Operations
    vector(const self& that){   //that Konstruktor
    
//     std::cout<< "Konstruktor\n";
	dim= that.dim;
	start= new T[dim];
	on_host= that.on_host;
	if (on_host)
	    for (int i= 0; i < dim; i++)
		start[i]= that.start[i];
	else
	    cudaMemcpy(dptr, that.dptr, dim*sizeof(T), cudaMemcpyDeviceToDevice);
    }

    self& operator=(const self& that)
    {
//	std::cout<< "x= y zuweisung Vector-Vector\n";
	assert(dim == that.dim);
	if (this != &that) {  //unnoetige Zuweisung vermeiden
	    on_host= that.on_host;
	    if (on_host) {
		for (int i= 0; i < dim; i++)
		    start[i]= that.start[i];
	    } else {
		cudaMemcpy(dptr, that.dptr, dim*sizeof(T), cudaMemcpyDeviceToDevice);
	    }
	}
	return *this;
    }

    using assign_base::operator=;

    void change_dim(size_type n) { throw "Not implemented yet! \n"; }

#if 0
    self operator + (const self &v1) 
    {   
	self temp(dim,0);
	temp=*this;
	assert(temp.dim == v1.dim);
	on_host= v1.on_host;
	 if (on_host) {
	     for (int i= 0; i < v1.dim; i++)
		 temp[i]+= v1.start[i];
	 } else  {
	    to_device(); // if not yet there
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE); 
	    vector_vector_rplus<<<dimGrid, dimBlock>>>(temp.dptr, v1.dptr, dim);
	 }
	 return temp;
    }

    void plus(const self& v_in, self& v_out)
    {
	assert(v_in.dim == dim && v_out.dim == dim);
	if (meet_data(*this, v_in, v_out)) {
	    for (int i= 0; i < dim; i++)
		 v_out[i]= start[i] + v_in.start[i];
	 } else  {
	     v_out.to_device();
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE); 
	    vector_vector_assign_plus<<<dimGrid, dimBlock>>>(v_out.dptr, dptr, v_in.dptr, dim);
	 }	
    }



    self operator - (const self &v1) 
    {   
	self temp(dim,0);
	temp=*this;
	assert(temp.dim == v1.dim);
	on_host= v1.on_host || this->on_host;
	 if (on_host) {
	     for (int i= 0; i < v1.dim; i++)
		 temp[i]-= v1.start[i];
	 } else  {
	    to_device(); // if not yet there
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE); 
            vector_vector_rminus<<<dimGrid, dimBlock>>>(temp.dptr, v1.dptr, dim);

	 }
	 return temp;
    }




    self operator * (const self &v1) 
    {   
	self temp(dim,0);
	temp=*this;
	assert(temp.dim == v1.dim);
	on_host= v1.on_host;
	 if (on_host) {
	     for (int i= 0; i < v1.dim; i++)
		 temp[i]*= v1.start[i];
	 } else  {
	    to_device(); // if not yet there
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE);  
            vector_vector_rmult<<<dimGrid, dimBlock>>>(temp.dptr, v1.dptr, dim);

	 }
	 return temp;
    }
 
 #if 0
 
     self operator / (const self &v1) 
    {   
	self temp(dim,0);
	temp=*this;
	assert(temp.dim == v1.dim);
	on_host= v1.on_host;
	 if (on_host) {
	     for (int i= 0; i < v1.dim; i++)
		 temp[i]-= v1.start[i];
	 } else  {
	    to_device(); // if not yet there
	    dim3 dimGrid(1), dimBlock(dim); 
            vector_vector_rdivide<<<dimGrid, dimBlock>>>(temp.dptr, v1.dptr, dim);

	 }
	 return temp;
    }
 
#endif





    //Scalar operations with vector
    // Expensive !!!
    template <typename U>
    self& operator=(const vector<U>& that)
    {  
 //       std::cout<<"Vector<U>-Vector Operator =\n";
	that.replicate_on_host();
	on_host= true;
	for (int i= 0; i < dim; i++)
	    start[i]= that.start[i];
	return *this;
    }


    template <typename U>
    self& operator=(const U& src)
    {	
//	std::cout<< "x=wert zuweisung\n";
        for (int i= 0; i < dim; i++) 
            start[i]= src;
	if (!on_host) { on_host= true; to_device(); }
	return *this;
    }


    template <typename U>
    self& operator*=(const U& src)
    {
 //       std::cout<< "x*= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    //std::cout<< "on host\n";
	    for (int i= 0; i < dim; i++) 
		start[i]*= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE); 
	    vec_rmult_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }

    template <typename U>
    self& operator/=(const U& src)
    {
 //       std::cout<< "x/= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    //std::cout<< "on host\n";
	    for (int i= 0; i < dim; i++) 
		start[i]/= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE);
	    vec_rdivide_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }


    template <typename U>
    self& operator+=(const U& src)
    {
 //       std::cout<< "x+= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    for (int i= 0; i < dim; i++) 
		start[i]+= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE);
	    vec_rplus_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }

    template <typename U>
    self& operator-=(const U& src)
    {
 //      std::cout<< "x-= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    for (int i= 0; i < dim; i++) 
		start[i]-= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(gridDimx(dim)), dimBlock(BLOCK_SIZE);
	    vec_rminus_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }
#endif

    T& operator[](int index) {
//	std::cout<<"klammer function 1\n\n";
        assert(index >= 0 && index < dim);
	to_host();
	return start[index];
    }


    T read(int i) const 
    {
        assert(i >= 0 && i < dim);
	return on_host ? start[i] : get_device_value(dptr + i);
    }

    T operator[](int i) const { 
 //     std::cout<<"klammer function 2\n\n"; 
      return read(i); 
    }

    T operator()(int i) const { return this->operator[](i); }
    T& operator()(int i) { return this->operator[](i); }

    __device__ T dat(int i) const { return dptr[i]; }
    __device__ T* dadd(int i) { return dptr + i; }

    bool valid_host() const { return on_host; }
    bool valid_device() const { return !on_host; }
    friend size_type size(const self& x) { return x.dim; }

    void to_host() const
    {
	if (!on_host) {
	    cudaMemcpy(const_cast<self*>(this)->start, dptr, sizeof(T)*dim, cudaMemcpyDeviceToHost);
	    const_cast<self*>(this)->on_host= true;
	}
    }

    void replicate_on_host() const
    {
	if (!on_host) {
	    cudaMemcpy(const_cast<self*>(this)->start, dptr, sizeof(T)*dim, cudaMemcpyDeviceToHost);
	}
    }

    void to_device() const
    {
	if (on_host) {
	    cudaMemcpy(const_cast<self*>(this)->dptr, start, sizeof(T)*dim, cudaMemcpyHostToDevice);
	    const_cast<self*>(this)->on_host= false;
	}
    }



   int gridDimx(int dim){
       int gridDimx= (dim/BLOCK_SIZE+1);
       
       if(gridDimx<65535) return gridDimx;
       else return 65535;
   
   }
   

    T* get_device_pointer() { return dptr; }
    const T* get_device_pointer() const { return dptr; }

    friend std::ostream& operator<<(std::ostream& os, const self& x)
    {
	
	x.replicate_on_host();
	os << "{" << size(x) << (x.valid_host() ? ",host}[" : ",device}[");

#if MTL_SHORT_PRINT
	bool complete= size(x) <= 20;
#else
	bool complete= true;
#endif

	if(complete)
	    for (int i= 0; i < size(x); i++)
		os << x.start[i] << (i < x.dim - 1 ? ", " : "]");
	else {
	    for (int i= 0; i < 10; i++)
		os << x.start[i] << ", ";
	    if(size(x)> 20) {
		os << "... ,";
		for (int i= size(x)-10; i < size(x); i++)
		    os << x.start[i] << (i < x.dim - 1 ? ", " : "]");
	    }
	}
	os << "\n";
	return os;
    }

  
    int  dim;
    T*   start; // Value on host //TO DO    malloc sizeof(T)*dim
    T*   dptr;   // Value on device (allocated as pointer whose content is referred)
    bool on_host;
    
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_VECTOR_INCLUDE
