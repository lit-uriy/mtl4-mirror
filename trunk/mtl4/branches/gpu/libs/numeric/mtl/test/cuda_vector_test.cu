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

#include <boost/numeric/mtl/cuda/vector_cuda.cu>

template <typename T>
void test(const char* name)
{
    typedef mtl::cuda::vector<T>   vt;

    std::cout << name << "-- Vector Test\n"; 
    mtl::cuda::vector<T>  x(3, 33), y(3, 10, false), z(3, 2), a(3,0);
    std::cout << "Vector constructed.\n" << "x=" << x << "\n";
    std::cout << "Vector constructed.\n" << "y=" << y << "\n";

    x.to_host();
    x.to_device();

    x= 4.0;
    x.to_device();
    if (x[0] != T(4))
	std::cout<< "Error assign vector on device.";
    
    std::cout << "const x[1] == " << x[1] << '\n';
    x[1]= 22;
    x.to_device();
    std::cout<< "x=" << x << "\n";
  
    y= x;           // Copy on device
    if (y[1] != T(22))
	std::cout<< "Error copy vector on device.";
    std::cout<< "y=" << y << "\n";
    
    x.to_device();
    x*= 7;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(28))
	std::cout<< "Error multipliying vector with scalar on device.";
    
    x.to_device();
    x+= 2;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(30))
	std::cout<< "Error adding vector with scalar on device.";
    
    x.to_device();
    x-= 10;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(20))
	std::cout<< "Error subtract vector with scalar on device.";
    
    x.to_device();
    x/= 10;
    std::cout<< "x=" << x << "\n";
    if (x[0] != T(2))
	std::cout<< "Error divide vector with scalar on device.";
    
}


int main(int argc, char* argv[])
{
    using namespace mtl;

    test<int>("int");
    test<short>("short");

    //test<char>("char"); // works but annoying print outs :-/
    test<float>("float");
    test<double>("double");

#if 0 // CUDA is too dumb for complex
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");
#endif

    return 0;
}
