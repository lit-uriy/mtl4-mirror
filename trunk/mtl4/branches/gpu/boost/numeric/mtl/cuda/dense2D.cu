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

#ifndef MTL_CUDA_DENSE2D_INCLUDE
#define MTL_CUDA_DENSE2D_INCLUDE

#include <iostream>
#include <cassert>
#include <string.h>

#include <boost/numeric/mtl/cuda/config.cu>
#include <boost/numeric/mtl/cuda/device_dense2D_new.cu>
#include <boost/numeric/mtl/cuda/dense2D_kernel.cu>
#include <boost/numeric/mtl/cuda/get_device_value.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>

#define BLOCK_SIZE 512


namespace mtl { namespace cuda {

/// Class for replicating dense matrix on host and device
template <typename T>
class dense2D
{
    typedef dense2D<T>                self;
  public:
    typedef T                        value_type;

    /// Constructor from type T 
    dense2D(int num_rows=1, int num_cols=1, const T& value= T() , bool on_host=true ) 
      : 
      num_rows(num_rows),
      num_cols(num_cols), 
      start(new T* [num_rows]),
      dptr(device_dense2D_new<T>(num_rows, num_cols)),
      on_host(on_host) 
    { // Matrix Allocation
      for(int i= 0; i < num_rows; i++){
	start[i] =  new T [num_cols];  // start[i] =  tmp + i * num_cols;
	memset(start[i], 0x0, sizeof(T) * num_cols);
      }

    } 

    ~dense2D() {
      // Deallocation
 	for(int i= num_rows-1; i>=0; i--)
 	    delete [] start[i];
 	delete [] start;
	cudaFree(dptr);
    }

#if 0
//Matrix-Matrix Operations
    dense2D(const self& that){   //that Konstruktor
	 std::cout<< "that Konstruktor\n";
	num_cols= that.num_cols;
	num_rows= that.num_rows;
	start = new T* [num_rows];
	for(int i=0; i  < num_rows; i++)
	  start[i] = new T [num_cols];
	
	
	on_host= that.on_host;
	if (on_host){
	    for (int i= 0; i < num_rows; i++){
		for (int j= 0; j < num_cols; j++){
		    start[i][j]= that.start[i][j];
		}
	    }	
	} else {
	    cudaMemcpy(dptr, that.dptr, num_cols*num_rows*sizeof(T), cudaMemcpyDeviceToDevice);
	}
    }


  self& operator=(const self& that)
    {
	std::cout<< "x= y zuweisung\n";
	assert((num_cols == that.num_cols) && (num_rows == that.num_rows));
        
	if (this != &that) {  //unnoetige Zuweisung vermeiden
	    on_host= that.on_host;
	    if (on_host) {
		for (int i= 0; i < num_rows; i++){
		    for (int j= 0; j < num_cols; j++){
		      start[i][j]= that.start[i][j];
		    }
		}	
	    } else {
		  std::cout<< "x= y zuweisung device\n";
		    for(int i=0; i < num_cols*num_rows; i++)
		    cudaMemcpy(dptr +i, that.dptr +i , sizeof(T), cudaMemcpyDeviceToDevice);
		    
		    std::cout<< "x= y zuweisung device ready\n";
	    }
	}
	return *this;
    }

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
	    dim3 dimGrid(dim/BLOCK_SIZE), dimBlock(BLOCK_SIZE); 
	    vector_vector_rplus<<<dimGrid, dimBlock>>>(temp.dptr, v1.dptr, dim);
	 }
	 return temp;
    }


    self operator - (const self &v1) 
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
	    dim3 dimGrid(dim/BLOCK_SIZE), dimBlock(BLOCK_SIZE); 
            vector_vector_rminus<<<dimGrid, dimBlock>>>(temp.dptr, v1.dptr, dim);

	 }
	 return temp;
    }

    //Scalar operations with vector
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
	//std::cout<< "x=wert zuweisung\n";
        for (int i= 0; i < dim; i++) 
            start[i]= src;
	if (!on_host) { on_host= true; to_device(); }
	return *this;
    }

    template <typename U>
    self& operator*=(const U& src)
    {
        //std::cout<< "x*= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    //std::cout<< "on host\n";
	    for (int i= 0; i < dim; i++) 
		start[i]*= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(dim/BLOCK_SIZE), dimBlock(BLOCK_SIZE); 
	    vec_rmult_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }

    template <typename U>
    self& operator/=(const U& src)
    {
        //std::cout<< "x*= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    //std::cout<< "on host\n";
	    for (int i= 0; i < dim; i++) 
		start[i]/= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(dim/BLOCK_SIZE), dimBlock(BLOCK_SIZE);
	    vec_rdivide_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }


    template <typename U>
    self& operator+=(const U& src)
    {
        //std::cout<< "x*= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    for (int i= 0; i < dim; i++) 
		start[i]+= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(dim/BLOCK_SIZE), dimBlock(BLOCK_SIZE);
	    vec_rplus_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }

    template <typename U>
    self& operator-=(const U& src)
    {
        //std::cout<< "x*= wert zuweisung\n";
	if (on_host && dim < host_limit) {
	    for (int i= 0; i < dim; i++) 
		start[i]-= src;
	} else {
	    to_device(); // if not yet there
	    dim3 dimGrid(dim/BLOCK_SIZE), dimBlock(BLOCK_SIZE);
	    vec_rminus_asgn<value_type> sc(src, dptr, dim);
	    launch_function<<<dimGrid, dimBlock>>>(sc);
	}
        return *this;
    }

#endif

    self& operator()(T scr, int num_row, int num_col) 
    {   
	assert(num_row >= 0 && num_row < num_rows && num_col >= 0 && num_col < num_cols);
	start[num_row][num_col]= scr;  //set on host
	
	int tmp=num_row*num_rows + num_col;

	cudaMemcpy(dptr+ tmp , &scr, sizeof(T), cudaMemcpyHostToDevice);  //set on device
	
	return *this;
    }

    T& operator()(int num_row, int num_col) {
	assert(num_row >= 0 && num_row < num_rows && num_col >= 0 && num_col < num_cols);	
	to_host();
	return start[num_row][num_col];
    }

    T read(int num_row, int num_col) const 
    {
        assert(num_row >= 0 && num_row < num_rows && num_col >= 0 && num_col < num_cols);
	return on_host ? start[num_row][num_col] : get_device_value(dptr + num_row* num_rows + num_col) ;
    }

    T operator()(int num_row, int num_col) const { return read(num_row, num_col); }

    bool valid_host() const { return on_host; }
    bool valid_device() const { return !on_host; }
    friend int  num_cols(const self& x) { return x.num_cols; }
    friend int  num_rows(const self& x) { return x.num_rows; }
    friend int  size(const self& x) { return x.num_rows * x.num_cols; }

    template<typename Vector>
    Vector operator * (const Vector& x)
    {	
	assert(num_rows >= 0 && num_rows == size(x));
	Vector tmp(size(x), 0);
 	tmp.on_host= !(x.valid_device() && (*this).valid_device());
	if (tmp.on_host){
	    for (int i= 0; i < size(x); i++){
		for (int j= 0; j < size(x); j++){
		    tmp[i]+= start[i][j]* x[j];
		}
	    }
	    
	} else {
	    tmp.to_device(); // if not yet there
	    dim3 dimGrid(num_cols/BLOCK_SIZE+1), dimBlock(BLOCK_SIZE);
	    mat_vec_mult<value_type, value_type><<<dimGrid, dimBlock>>>(tmp.dptr, dptr, x.dptr, num_rows, num_cols);
	}
	return tmp;
    }

    void set_to_zero(bool set_on_host= false) 
    {  
	on_host= set_on_host;
	start[0][0]= T(0);
	
	if(on_host == true)
	    for(int i= 0; i < num_rows; i++)
		memset(start[i], 0x0, sizeof(T) * num_cols);
	    
	
	if(on_host == false){

// 	copy value to first entry and replicate it
//	cudaMemcpy(dptr, &start[0][0], sizeof(T), cudaMemcpyHostToDevice);
//	cudaMemcpy(dptr + 1, dptr, (num_cols*num_rows-1) * sizeof(T), cudaMemcpyDeviceToDevice);


	cudaMemcpy(dptr , &start[0][0], sizeof(T), cudaMemcpyHostToDevice);
	for (int i= 1; i < num_cols; i++){
	    cudaMemcpy(dptr + i, dptr, sizeof(T), cudaMemcpyDeviceToDevice);
	} //first Line is zero
	for (int i= 1; i < num_rows; i++){
	      cudaMemcpy(dptr + num_cols*i, dptr, sizeof(T)*num_cols, cudaMemcpyDeviceToDevice);
	}

	
	}
	
    }

   

   /// laplacian setup on host
   void laplacian_setup_host(T d)
   {
	bool tmp=valid_device();
	
        if(tmp){
	    to_host(); 
	    on_host=true;
	}
        
//	set_to_zero():  //i'm not sure, if we need this 

	for(int i=0; i<num_rows; i++){
	    
            if(i==0){
		 start[i][i]=d;
		 start[i][i+1]=-1;
	    }
	    
	    if(i>0 && i<num_rows-1){
		start[i][i-1]=-1;
		start[i][i]=d;
		start[i][i+1]=-1;
	    }
	    
	    if(i==num_rows-1){
		start[i][i-1]=-1;
		start[i][i]=d;
		
	    }
	    
	}
	
    if(tmp) to_device(); 
	
   }
   

  /// laplacian setup on device
   void laplacian_setup_device(T d)
   {
       
     bool tmp=valid_host();  
     
     if(tmp){
	    //	set_to_zero():  //i'm not sure, if we need this 
	    to_device(); 
	    on_host=false;
	}
       
     dim3 dimGrid(num_cols/BLOCK_SIZE+1), dimBlock(BLOCK_SIZE);
     laplacian<value_type, value_type><<<dimGrid, dimBlock>>>(dptr, d, num_rows);
     
     if(tmp) to_host();
     
   } 
   
     
   void to_host() const
    {
	if (!on_host) {
	    int temp= 0;
	    for (int i= 0; i < num_rows; i++){
		for (int j= 0; j < num_cols; j++){
		    cudaMemcpy(&(const_cast<self*>(this)->start[i][j]), (dptr + temp), sizeof(T), cudaMemcpyDeviceToHost);
		    temp++;
		}
	    }
	 const_cast<self*>(this)->on_host= true;   
	}
    }

    void replicate_on_host() const
    {
	if (!on_host) {
	    int temp= 0;
	    T aux;
	    for (int i= 0; i < num_rows; i++){
		for (int j= 0; j < num_cols; j++){
		    cudaMemcpy(&aux, (dptr + temp), sizeof(T), cudaMemcpyDeviceToHost);
		    cudaMemcpy(&start[i][j], (dptr + temp), sizeof(T), cudaMemcpyDeviceToHost);
		    temp++;
		}
	    }
	}
    }

    void to_device() const
    {
	if (on_host) {
	    int temp= 0;
	    for (int i= 0; i < num_rows; i++){
		for (int j= 0; j < num_cols; j++){
		    cudaMemcpy((const_cast<self*>(this)->dptr + temp), &start[i][j], sizeof(T), cudaMemcpyHostToDevice);
		    temp++;
		}
	    }
	    const_cast<self*>(this)->on_host= false;
	}
    }
    
    T* get_device_pointer() { return dptr; }
    const T* get_device_pointer() const { return dptr; }

    friend std::ostream& operator<<(std::ostream& os, const self& x)
    {
	x.replicate_on_host();
	os << "{" << x.num_rows << "," << x.num_cols << (x.valid_host() ? ",host}=\n" : ",device}=\n");
	for (int i= 0; i < x.num_rows; i++){
	os << "[ ";  
	  for (int j= 0; j < x.num_cols; j++){
	     os <<  x.start[i][j] << (j== x.num_cols-1 ? " ]\n" : " ");	  
	  }
	}
	 os << "\n"; 
	   
	return os;
    }

    int  num_cols, num_rows;
    T**   start; 
    T*   dptr;   // Value on device (allocated as pointer whose content is referred)
    bool on_host;
    
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_DENSE2D_INCLUDE
