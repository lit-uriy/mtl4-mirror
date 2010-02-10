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
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

#include <boost/numeric/mtl/cuda/device_new.hpp>
#include <boost/numeric/mtl/cuda/vector_cuda.hpp>

template <typename T>
void test(const char* name)
{
    typedef mtl::cuda::vector<T>   vt;

    std::cout << name << "Vector Test\n"; 
    mtl::cuda::vector<T>  x(3, 33), y(3, 10, false);//, z(3,0), a(3,0);
    std::cout << "Vector constructed.\n" << "x=" << x << "\n";
    std::cout << "Vector constructed.\n" << "y=" << y << "\n";

    x.to_host();
    x.to_device();

    x= 4.0;
    x.to_device();
    std::cout << "Vector assigned.\n" << "x=" << x << "\n";

    std::cout << "const x[1] == " << x[1] << '\n';

    x.to_device();
    std::cout<< "x=" << x << "\n";
   
    std::cout << "whole vector assigned on device.\n"; 

    std::cout<< "vorher y=" << y << "\n";
    y= x;           // Copy on device

    std::cout<< "y=" << y << "\n";
    x.to_device();
    x*= 2;
    std::cout<< "x=" << x << "\n";
#if 0 
   x.init(23);
    x.to_device();
    y= x;
    std::cout<< "y=" << y << "\n";
    x.init(12);
    x.to_host();   
    std::cout<< "z=" << z << "\n";
    z= x;
    std::cout<< "z=" << z << "\n";  


    if (y.value() != T(4))
	throw "Error copying scalar on device.";
    print('x', x);


    x*= 2;   // Computing on device
    print('x', x);
    if (!x.valid_device())
	throw "No valid copy on device.";
    print('x', x);
    if (x.value() != T(10))
	throw "Error computing  device.";
     x[1]= 100;
    std::cout << "component of vector assigned.\n";
#endif
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;

    test<int>("int");
#if 0
    test<short>("short");

    test<char>("char");
    test<float>("float");
    test<double>("double");
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");
#endif
    return 0;
}
