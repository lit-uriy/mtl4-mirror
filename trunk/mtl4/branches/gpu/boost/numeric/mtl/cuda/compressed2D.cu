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
#include <boost/numeric/mtl/cuda/compressed2D_kernel.cu>
#include <boost/numeric/mtl/cuda/get_device_value.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>
#include <boost/numeric/mtl/cuda/meet_data.cu>

#define BLOCK_SIZE 512


namespace mtl { namespace cuda {
/// Class for replicating dense matrix on host and device
template <typename T>
class compressed2D
{
    typedef compressed2D<T>                self;

  public:
    typedef T                        value_type;
    typedef unsigned                 size_type;

    /// Constructor from type T 
    compressed2D(unsigned num_rows=1, unsigned num_cols=1, const T& value= T() , bool on_host=true )
      : nnz(0),
	num_rows(num_rows),
        num_cols(num_cols),
	h_ptr(new unsigned[num_rows+1]), 
	h_indices(0), 
	h_data(0),
	d_ptr(device_vector_new<unsigned>(0, num_rows+1)),
	d_indices(device_vector_new<unsigned>(0, 0)),
	d_data(device_vector_new<T>(T(0), 0)),
        on_host(on_host)
    {
      for (unsigned i= 0; i <= num_rows; i++) h_ptr[i]= 0;
    }

    ~compressed2D() {
      // Deallocation
        delete [] h_ptr;
	delete [] h_indices;
        delete [] h_data;
	cudaFree(d_ptr);
        cudaFree(d_indices);
        cudaFree(d_data);
    }


    T& operator()(int num_row, int num_col) {
        assert(num_row >= 0 && num_row < num_rows && num_col >= 0 && num_col < num_cols);
        to_host();
        return h_data[num_row * num_rows + num_col];
    }

    T read(int row, int col) const
    {
        assert(row >= 0 && row < num_rows && col >= 0 && col < num_cols);
        T temp;
        int num_elm;
        num_elm= h_ptr[row+1]-h_ptr[row];
//       std::cout<< "row=" << row << "  col=" << col << " num_elm=" << num_elm << "\n";
        for (unsigned i= h_ptr[row]; i < h_ptr[row]+num_elm; i++){
// 		std::cout<< "schleife i=" << i << "\n";
                if (h_indices[i] == col && on_host)
                {
                        temp= h_data[i];
// 			std::cout<< "temp=" << temp << "\n";
			break;
                }
                else if (h_indices[i] == col && !on_host)
                {
                        temp= get_device_value(&d_data[i]);
			break;
                }
                else
                {
                        temp= T(0);
                }
        }
        return temp;
    }

    T operator()(int num_row, int num_col) const { return read(num_row, num_col); }

    bool valid_host() const { return on_host; }
    bool valid_device() const { return !on_host; }
    friend int  num_cols(const self& x) { return x.num_cols; }
    friend int  num_rows(const self& x) { return x.num_rows; }
    friend int  size(const self& x) { return x.num_rows * x.num_cols; }

    template<typename Vector>
    Vector operator* (const Vector& x)
    {	
	assert(num_cols == size(x));
	Vector tmp(num_rows, 0);
	
	if (meet_data(*this, x, tmp)) {
	   for (size_type row= 0; row < num_rows; ++row) {
	       value_type sum(0);
	       for (size_type start= h_ptr[row], end= h_ptr[row+1]; start != end; ++start)
		   sum+= h_data[start] * x[h_indices[start]];
	       tmp[row]= sum;
	   }
	} else {
	    dim3 dimGrid(num_cols/BLOCK_SIZE+1), dimBlock(BLOCK_SIZE);
	    sparse_mat_vec_mult<<<dimGrid, dimBlock>>>(num_rows, d_ptr, d_indices, d_data, x.dptr, tmp.dptr);	    
	}
	return tmp;
    }

    void change_nnz(unsigned n)
    {
        nnz= n;
	if (h_indices) delete[] h_indices;
        h_indices= new unsigned [n];
	if (h_data) delete[] h_data;
        h_data= new T [n];

        d_ptr= device_vector_new<unsigned>(0, num_rows+1);
	if (d_indices) cudaFree(d_indices);
        d_indices= device_vector_new<unsigned>(0, n);
	if (d_data) cudaFree(d_data);
        d_data= device_vector_new<T>(T(0), n);
    }

    void laplacian_setup(unsigned m, unsigned n)
    {
	assert(m*n == num_rows); assert(num_cols == num_rows);
	change_nnz(5*m*n - 2*m - 2*n);
	
	unsigned pos= 0;
	for (unsigned i= 0; i < m; i++)
	      for (unsigned j= 0; j < n; j++) {
		  unsigned row= i * n + j;
		  h_ptr[row]= pos;
		  if (i > 0) {
		     h_indices[pos]= row-n;
		     h_data[pos++]= -1;
		  }
		  if (j > 0) {
		     h_indices[pos]= row-1;
		     h_data[pos++]= -1;
		  }
		  h_indices[pos]= row;
		  h_data[pos++]= 4;
		  if (j < n-1)  {
		     h_indices[pos]= row+1;
		     h_data[pos++]= -1;
		  } 
		  if (i < m-1) {
		     h_indices[pos]= row+n;
		     h_data[pos++]= -1;
		  }
	      }
        assert(pos == 5*m*n - 2*m - 2*n);
	h_ptr[num_rows]= pos;
	on_host= true;
    }


    void simpel_laplacian_setup(unsigned n, int d)
    {
        unsigned num=n+2*(n-1), temp=0;
        change_nnz(num);
// 	std::cout<< "n=" << n << " laplace elemente=" << num << "\n";
//         std::cout<< "simpel_laplacian_setup\n";
        h_ptr[0]= 0; 
	h_ptr[1]= 2; 
	h_indices[0]= 0; 
	h_indices[1]= 1;
        h_data[0] = T(d); 
	h_data[1] = T(-1);
	
        for (unsigned i= 2; i < num-4; i+=3) {
          h_indices[i]=  temp; 
	  h_indices[i+1]= temp+1; 
	  h_indices[i+2]= temp+2;
          h_data[i]=   T(-1); 
	  h_data[i+1]= T(d); 
	  h_data[i+2]= T(-1);
          temp++;
     //     h_ptr[temp+1]=h_ptr[temp]+3;
        }
	for (unsigned i= 2; i < num_rows+1; i++) {
// 	  std::cout << "i=" << i << "\n";
	  h_ptr[i]=h_ptr[i-1]+3;
	}
	h_ptr[num_rows]= h_ptr[num_rows-1]+2; 
	h_indices[num-2]= n-2; 
	h_indices[num-1]= n-1;
        h_data[num-1] = T(d); 
	h_data[num-2] = T(-1);

	//sending to device
        cudaMemcpy(d_ptr, h_ptr, sizeof(unsigned)*(num_rows+1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, h_indices, sizeof(unsigned)*(num), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data, h_data, sizeof(T)*(num), cudaMemcpyHostToDevice);
// for (unsigned i= 0; i < num_rows+1; i++)
//         std::cout<< "i=" << i << "h_ptr[i]=" << h_ptr[i] << "\n";
// 
// for (unsigned i= 0; i < num; i++){
//         std::cout<< "i=" << i << "  h_data[i]=" << h_data[i] <<  "  i=" << i << "h_indives[i]=" << h_indices[i] << "\n";
// }
	
// 	 std::cout<< "\nData:[ ";
// 	for (unsigned i= 0; i < num; i++) {
//                std::cout<<  h_data[i] << (i==num-1 ? " " : ", ");
//         }
// 	std::cout<< "]\nCols:[ ";
// 	for (unsigned i= 0; i < num; i++) {
//                std::cout<< h_indices[i] << (i==num-1 ? " " : ", ");
//         }	
//         std::cout<< "]\nptr :[ ";
// 	for (unsigned i= 0; i < num_rows+1; i++) {
//                std::cout<< h_ptr[i] << (i==num-1 ? " " : ", ");
//         }
//         std::cout<< "]\n\n";

    }
 
    void set_to_zero()
    {
	change_nnz(1);
	std::cout<< " num_rows=" << num_rows << "\n";
        for (unsigned i= 0; i < num_rows+1; i++){       //initializing on host 
            h_ptr[i]= 0;				//sparse rows pointer
        }
	std::cout<< " nnz=" << nnz << "\n";
        for (unsigned i= 0; i < nnz; i++){	       
            h_data[i]= 0;				//initializing data
            h_indices[i]= T(0);				//initializing cols
        }
	std::cout<< " dim2=" << nnz << "\n";
        on_host= false;
   
	//wesentlich schneller                          //initializing on device
        cudaMemcpy(d_ptr , &h_ptr[0], sizeof(unsigned), cudaMemcpyHostToDevice);
        for (int i= 1; i < num_rows+1; i++){
            cudaMemcpy(d_ptr + i, d_ptr, sizeof(unsigned), cudaMemcpyDeviceToDevice);
        }
        cudaMemcpy(d_indices , &h_indices[0], sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data , &h_data[0], sizeof(T), cudaMemcpyHostToDevice);
        for (int i= 1; i < nnz; i++){
            cudaMemcpy(d_indices + i, d_indices, sizeof(unsigned), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_data + i, d_data, sizeof(T), cudaMemcpyDeviceToDevice);
        }

    }

    void to_host() const
    {
        if (!on_host) {
            cudaMemcpy(h_ptr, d_ptr, sizeof(unsigned)*(num_rows+1), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_indices, d_indices, sizeof(unsigned)*(nnz), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_data, d_data, sizeof(T)*(nnz), cudaMemcpyDeviceToHost);
            const_cast<self*>(this)->on_host= true;
        }
    }

    void replicate_on_host() const
    {
        if (!on_host) {
           cudaMemcpy(h_ptr, d_ptr, sizeof(unsigned)*(num_rows+1), cudaMemcpyDeviceToHost);
           cudaMemcpy(h_indices, d_indices, sizeof(unsigned)*(nnz), cudaMemcpyDeviceToHost);
           cudaMemcpy(h_data, d_data, sizeof(T)*(nnz), cudaMemcpyDeviceToHost);
        }
    }
 
    void to_device() const
    {
        if (on_host) {
           cudaMemcpy(d_ptr, h_ptr, sizeof(unsigned)*(num_rows+1), cudaMemcpyHostToDevice);
           cudaMemcpy(d_indices, h_indices, sizeof(unsigned)*(nnz), cudaMemcpyHostToDevice);
           cudaMemcpy(d_data, h_data, sizeof(T)*(nnz), cudaMemcpyHostToDevice);
           const_cast<self*>(this)->on_host= false;
        }
    }

    T* get_device_pointer() { return d_ptr; }
    const T* get_device_pointer() const { return d_ptr; }

    friend std::ostream& operator<<(std::ostream& os, const self& x)
    {
        x.replicate_on_host();
        os << "{" << x.num_rows << "," << x.num_cols << (x.valid_host() ? ",host}=\n" : ",device}=\n");
        for (int i= 0; i < x.num_rows; i++){
        os << "[";
          for (int j= 0; j < x.num_cols; j++){
             os <<  x.read(i,j) << (j==x.num_cols-1 ? "]\n" : "\t");
          }
        }
         os << "\n";

	 
	 
/*	unsigned  num=x.num_cols+2*(x.num_cols-1);
	 os<< "\nData:[ ";
	for (unsigned i= 0; i < num; i++) {
               os<<  x.h_data[i] << (i==num-1 ? " " : ", ");
        }
	os<< "]\nCols:[ ";
	for (unsigned i= 0; i < num; i++) {
               os<< x.h_indices[i] << (i==num-1 ? " " : ", ");
        }	
        os<< "]\nptr :[ ";
	for (unsigned i= 0; i < x.num_rows+1; i++) {
               os<< x.h_ptr[i] << (i==num-1 ? " " : ", ");
        }
        std::cout<< "]\n\n";	 */
	 
        return os;
    }

    unsigned    nnz;
    unsigned    num_cols, num_rows;
    unsigned*   h_ptr;	   // sparse_rows on host
    unsigned*   d_ptr;	   // sparse_rows on device
    unsigned*   h_indices; // cols on host 
    unsigned*   d_indices; // cols on device
    T*     h_data;	// Value on host 
    T*     d_data;   	// Value on device (allocated as pointer whose content is referred)
    bool   on_host;

};

}} // namespace mtl::cuda

#endif // MTL_CUDA_COMPRESSED2D_INCLUDE
