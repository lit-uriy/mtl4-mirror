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

#ifndef MTL_CUDA_COMPRESSED2D_INCLUDE
#define MTL_CUDA_COMPRESSED2D_INCLUDE

#include <iostream>
#include <cassert>

#include <boost/numeric/mtl/cuda/config.cu>
#include <boost/numeric/mtl/cuda/device_dense2D_new.cu>
#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/dense2D_kernel.cu>
#include <boost/numeric/mtl/cuda/get_device_value.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>

#define BLOCK_SIZE 512


namespace mtl { namespace cuda {

/// Class for replicating dense matrix on host and device
template <typename T>
class compressed2D
{
    typedef compressed2D<T>                self;

  public:
    typedef T                        value_type;

    /// Constructor from type T 
    compressed2D(int num_rows=2,  int num_cols=0, const T& value= T() , bool on_host=true ) 
      : num_rows(num_rows),
        num_cols(num_cols),
      on_host(on_host) 
    {} 

    ~compressed2D() {
      // Deallocation
 	for(int i= 1; i>=0; i--)
 	    delete [] start[i];
 	delete [] start;
	delete [] sparse_rows;
	cudaFree(dptr);
    }

  
  template <typename U>
  self& compressed_matrix(const dense2D<U>& that){

    double index=0;
    that.to_host();
    
   
    num_rows=2;
    num_cols=that.elements(that);
    
    
    sparse_rows = new T [that.num_rows(that)];
    sparse_rows[0]=0;
    start = new T* [2];
    
    for(int i= 0; i < 2; i++)
	  start[i] = new T [that.elements(that)];
    
    for(int i= 0; i < that.num_rows(that); i++){
      if (i>0) sparse_rows[i]=sparse_rows[i-1];
      for(int j= 0; j < that.num_rows(that); j++) {
       if(that.start[i][j]!=0) {
        start[0][index]=that.start[i][j];
	start[1][index]=T(j);
	sparse_rows[i]++;
        index++;
       }
      }
    }
  return *this;
  }



//  for testing
  template <typename U>
  void prueba(const dense2D<U>& that){
  
    std::cout<<"Impresion de prueba, elementos de that= "<<that.elements(that);
   
    
  }





self& operator()(T scr, int num_row, int num_col) 
    {   
	assert(num_row >= 0 && num_row < num_rows && num_col >= 0 && num_col < num_cols);
	start[num_row][num_col]= scr;  //set on host
	
	int temp(num_row*num_rows + num_col);
	cudaMemcpy(dptr+ temp , &scr, sizeof(T), cudaMemcpyHostToDevice);  //set on device
	
	on_host= false;
		
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
    friend double  elements(const self& x) { return x.elements; } 
    

    template<typename Vector>
    Vector operator * (const Vector& x)
    {	
	assert(num_rows >= 0 && num_rows == size(x));
	Vector temp(size(x), 0);
 	temp.on_host= !(x.valid_device() && (*this).valid_device());
	if (temp.on_host){
	    for (int i= 0; i < size(x); i++){
		for (int j= 0; j < size(x); j++){
		    temp[i]+= start[i][j]*x[j];
		}
	    }
	} else {
	    temp.to_device(); // if not yet there
	    dim3 dimGrid(num_cols/BLOCK_SIZE+1), dimBlock(BLOCK_SIZE);
	    mat_vec_mult<value_type, value_type><<<dimGrid, dimBlock>>>(temp.dptr, dptr, x.dptr, num_rows, num_cols);
	}
	return temp;
    }

    void set_to_zero() 
    {
	start[0][0]= T(0);
	on_host= false;
	cudaMemcpy(dptr , &start[0][0], sizeof(T), cudaMemcpyHostToDevice);
	for (int i= 1; i < num_cols; i++){
	    cudaMemcpy(dptr + i, dptr, sizeof(T), cudaMemcpyDeviceToDevice);
	} //first Line is zero
	for (int i= 1; i < num_rows; i++){
	      cudaMemcpy(dptr + num_cols*i, dptr, sizeof(T)*num_cols, cudaMemcpyDeviceToDevice);
	}
	
    }

    void to_host() const
    {
	if (!on_host) {
	    int temp= 0;
	    for (int i= 0; i < num_rows; i++){
		for (int j= 0; j < num_cols; j++){
// 		    std::cout<< "i=" << i << "  ,j=" << j << "\n";
		    cudaMemcpy(&(const_cast<self*>(this)->start[i][j]), (dptr + temp), sizeof(T), cudaMemcpyDeviceToHost);
		    temp++;
		}
	    }
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
	os << "{2," << x.elements << (x.valid_host() ? ",host}=\n" : ",device}=\n");
	for (int i= 0; i < 2; i++){
	os << "[ ";  
	  for (int j= 0; j < x.elements; j++){
	     os <<  x.start[i][j] << (j== x.elements-1 ? " ]\n" : "\t");	  
	  }
	}
	 os << "\nSparse Rows\n["; 
	 
	 for (int i=0; i < (sizeof(x.sparse_rows) / sizeof(x.sparse_rows[0])); i++  )
	   os <<  x.sparse_rows[i] << (i== (sizeof(x.sparse_rows) / sizeof(x.sparse_rows[0])) ? " ]\n" : "\t");
	 
	 os << "\n"; 
	 
	return os;
    }

    int num_rows,  num_cols;

    double elements;
    int* sparse_rows;
    T**   start; // Matrix to save value and cols
    T*   dptr;   // Value on device (allocated as pointer whose content is referred)
    bool on_host;
    
};

}} // namespace mtl::cuda

#endif // MTL_CUDA_COMPRESSED2D_INCLUDE
