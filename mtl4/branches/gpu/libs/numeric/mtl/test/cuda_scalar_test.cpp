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
#include <boost/numeric/mtl/cuda/scalar.hpp>

template <typename T>
void inline print_position(const T& x)
{
    std::cout << "x.valid_device() == " << x.valid_device() << "x.valid_host() == " << x.valid_host();
    if (x.valid_device() == x.valid_host())
	throw "valid_device() and valid_host() must be different.";
}

template <typename T>
void inline print(const T& x)
{
    print_position(x);
    std::cout << "x is " << x << "\n";
}

template <typename T>
void test(const char* name)
{
    std::cout << name << '\n';
    
    mtl::cuda::scalar<T>  x, y;

    x= 3;
    print(x);

    x.to_device();
    print_position(x);
    if (!x.valid_device())
	throw "No valid copy on device.";

    x.to_device();
    print_position(x);
    if (!x.valid_host())
	throw "No valid copy on host.";

#if 0 
    x.to_device();
    x= 4;           // Setting on device
    print(x);
    if (x != T(4))
	throw "Error setting scalar on device.";

    x= 5;
    x.to_device();

    y= x;           // Copy on device
    print(y);
    if (y != T(5))
	throw "Error copying scalar on device.";

    if (!x.valid_device())
	throw "No valid copy on device.";
    x*= 2;                                   // Computing on device
    if (!x.valid_device())
	throw "No valid copy on device.";
    print(x);
    if (x != T(10))
	throw "Error computing  device.";
#endif
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;

    test<int>("int");
    test<short>("short");
    test<char>("char");
    test<float>("float");
    test<double>("double");
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");

    return 0;
}
