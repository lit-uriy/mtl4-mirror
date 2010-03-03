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

template <typename T>
void test(const char* name)
{
  
    std::cout << name << "000 Matrix Test\n"; 
    typedef mtl::cuda::dense2D<T>   dense;
    int size= 3;
    std::cout << name << "-- Matrix Test\n"; 
    mtl::cuda::dense2D<T>  x(size, size), y(size, size);//, z(3, 3), a(3,3);
     
    x.set_to_zero();
 
    std::cout << "Matrix constructed.\n" << "x=" << x << "\n";

            
    std::cout << "const x(0,0) == " << x(0,0) << '\n';
    x.to_host();
    for (int i= 0; i < size; i++){
      std::cout<< "i="  << x(i,i) << "\n";
      x=x(2,i,i);
    }
    std::cout << "const x(1,1) == " << x(1,1) << '\n';
    std::cout<< "x=" << x << "\n";
    y= x;           // Copy on device
    y.to_host();
    //if (y(0,0) != T(22))
//	std::cout<< "Error copy matrix on device.\n";
    
   // y.to_device();
    std::cout<< "y=" << y << "\n";
     std::cout<< "Hallo\n";
  
    #if 0  
    

    x.to_device();
    x*= 7;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(28))
	std::cout<< "Error multipliying matrix with scalar on device.\n";
    
    x.to_device();
    x+= 2;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(30))
	std::cout<< "Error adding matrix with scalar on device.\n";
    
    x.to_device();
    x-= 10;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(20))
	std::cout<< "Error subtract matrix with scalar on device.\n";
    
    x.to_device();
    x/= 10;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(2))
	std::cout<< "Error divide matrix with scalar on device.\n";
#endif
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
