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
    std::cout << "x.valid_device() == " << x.valid_device() 
	      << " x.valid_host() == " << x.valid_host() << '\n';
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
    std::cout << "Scalars constructed.\n";

    x= 3;
   
    print(x);
    x.to_device();
    x= 4;
    std::cout << "Scalars assigned.\n";
    print(x);

    // x.to_device();
    print_position(x);
    if (!x.valid_device())
	throw "No valid copy on device.";

    x.to_host();
    print_position(x);
    if (!x.valid_host())
	throw "No valid copy on host.";

    x.to_device();
    x= 7;           // Setting on device
    print(x);
    if (x.value() != T(7))
    	throw "Error setting scalar on device.";

    x= 5;
    x.to_device();

    y= x;           // Copy on device
    print(y);
    if (y.value() != T(5))
	throw "Error copying scalar on device.";
    if (!x.valid_device())
	throw "No valid copy on device.";
    print(x);

    std::cout<< "------------------------\n";
    x.to_device();   

#if 1 
    // std::cout << "x.h=" << x.hvalue << "\n";
    //std::cout << "x.d=" << *x.dptr << "\n";
    //std::cout << "x_on_host==" << x.on_host << "\n";
    x*= 2;   // Computing on device
    //std::cout << "_x.h=" << x.hvalue << "\n";
    //std::cout << "_x.d=" <<*x.dptr << "\n";
    //std::cout << "_x_on_host==" << x.on_host << "\n";
    print(x);
    std::cout << "x_loc-nach=" << x.valid_host() << "\n";
    if (!x.valid_device())
	throw "No valid copy on device.";
    print(x);
    if (x.value() != T(10))
	throw "Error computing  device.";
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
