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

#include <iostream>
#include <complex>
//#include <boost/test/minimal.hpp>
//#include <boost/numeric/mtl/mtl.hpp>

#include <boost/numeric/mtl/cuda/compressed2D.cu>
#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>



template <typename T>
void test(const char* name)
{
  
  
    typedef mtl::cuda::dense2D<T>   	dense2D;
    typedef mtl::cuda::compressed2D<T>   	compressed2D;
    typedef mtl::cuda::vector<T>  	vector;
    
    int size= 10;
    std::cout << name << "-- Matrix Test\n"; 
    dense2D  A(size, size);
    vector  x(size, 1.0), b(size, 0.0); 
    compressed2D mc;
     
    A.set_to_zero();
 
    std::cout << "Matrix constructed.\n";
    std::cout << "A.num_cols= "<< num_cols(A)<<"\nA.num_rows= " <<num_rows(A)<<"\nA.elements= " <<elements(A)<<"\n";
//  std::cout << "Matrix constructed.\n" << "A=" << A << "\n";


    for (int i= 0; i < size; i++){
      A(2,i,i);
    }
    A(2,0,size-1);
    A(2,size-1,0);
    
//    A(0,0,3); for testing "elemenst(A)"

    A.to_device();
    x.to_device();
    std::cout<< "A=" << A<<"\nA.elements= " <<elements(A)<<"\n";
    
//    std::cout<< "MC=" << mc<<"\nMC.elements= " <<elements(mc)<<"\n";
    
    
//    mc.compressed_matrix(A);
    mc.prueba(A);    

 
}


int main(int argc, char* argv[])
{
    using namespace mtl;

    cuda::activate_best_gpu();
    test<int>("int");
 //   test<short>("short");

    //test<char>("char"); // works but annoying print outs :-/
   // test<float>("float");
   // test<double>("double");

#if 0 // CUDA is too dumb for complex
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");
#endif

    return 0;
}
