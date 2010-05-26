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
    int m=182,n=182;
    int size= n*m;
    std::cout << name << "-- Matrix Test\n";
    mtl::cuda::compressed2D<T>  A(size, size);
    mtl::cuda::vector<T>  x(size, 1.0), b(size, 0.0);
    std::cout << "start to zero.\n" ;
    A.set_to_zero();
    std::cout << "end to zero.\n";
    
    
    std::cout << "Matrix constructed.\n";
//    std::cout << "A=" << A << "\n";
    A.laplacian_setup(m, n);
  
//     x[1]=2; x[2]= 3; x[3]= 4;

//    A.to_host();   x.to_host();   b.to_host();
    A.to_device(); x.to_device(); b.to_device();

    b= A * x;
    
//    std::cout<< "A=" << A << "\n";
//    std::cout<< "x=" << x << "\n";
//    std::cout<< "b=" << b << "\n";
    
    if (b[0] != T(2))
       std::cout<< "Error Matrix vector multiplication.\n";

    else std::cout<< "\nMatrix vector multiplication was Correct!!!.\n";  

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
