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
void inline print(char n, const T& x)
{
    std::cout << n << " is " << x << "\n";
}

template <typename T>
void test(const char* name)
{
    std::cout << name << "Vector Test\n";
    
    mtl::cuda::vector<T>  x(2), y(2);
    std::cout << "Vector constructed.\n";

    x= 4;
    std::cout << "whole vector assigned on host.\n";
    print('x', x);
    x.to_device();
    x= 5;
    std::cout << "whole vector assigned on devise.\n";

    y= x;           // Copy on device
    std::cout << "Vector copyed on device.\n";
    print('y', y);
    if (y.value() != T(4))
	throw "Error copying scalar on device.";
    print('x', x);

#if 0
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
    test<short>("short");
#if 0
    test<char>("char");
    test<float>("float");
    test<double>("double");
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");
#endif
    return 0;
}
