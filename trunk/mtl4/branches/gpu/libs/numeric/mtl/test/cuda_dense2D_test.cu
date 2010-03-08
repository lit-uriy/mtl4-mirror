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

#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>



template <typename T>
void test(const char* name)
{
  
    typedef mtl::cuda::dense2D<T>   dense;
    int size= 10;
    std::cout << name << "-- Matrix Test\n"; 
    mtl::cuda::dense2D<T>  A(size, size);
    mtl::cuda::vector<T>  x(size, 1.0), b(size, 0.0); 
     
    A.set_to_zero();
 
    //std::cout << "Matrix constructed.\n" << "A=" << A << "\n";
std::cout << "Matrix constructed.\n";
            
    std::cout << "const A(0,0) == " << A(0,0) << '\n';
    A.to_host();
    for (int i= 0; i < size; i++){
//       std::cout<< "i="  << A(i,i) << "\n";
      A(2,i,i);
    }
    A(2,0,size-1);
    A(2,size-1,0);
    std::cout << "const A(1,1) == " << A(1,1) << '\n';
//     std::cout<< "A=" << A << "\n";
     A.to_device();
     x.to_device();
      std::cout<< "A=" << A << "\n";
//      std::cout<< "x=" << x << "\n";
//      std::cout<< "b=" << b << "\n";
     b= A * x;
     std::cout<< "b=" << b << "\n";
     if (b[0] != T(4))
	std::cout<< "Error Matrix vector multiplication on device.\n";
     
  
 
}


int main(int argc, char* argv[])
{
    using namespace mtl;

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
