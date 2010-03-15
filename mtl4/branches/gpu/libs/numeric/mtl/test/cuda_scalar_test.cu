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


#include <boost/numeric/mtl/cuda/config.cu>
#include <boost/numeric/mtl/cuda/device_new.cu>
#include <boost/numeric/mtl/cuda/scalar.cu>

template <typename T>
void inline print(char n, const T& x)
{
    std::cout << n << " is " << x << "\n";
}

template <typename T>
void test(const char* name)
{
   
    std::cout << name << '\n';
    
    mtl::cuda::scalar<T>  x, y(2), z(0);
    std::cout << "Scalars constructed.\n";

    x= 4;
    std::cout << "Scalars assigned.\n";
    print('x', x);

    y= x;           // Copy on device
    print('y', y);
    if (y.value() != T(4))
	throw "Error copying scalar on device.";
    print('x', x);


    x*= 2;   // Computing on device
    print('x', x);
    if (x.value() != T(8))
	throw "Error computing *= on device.";

//    x/= z;   // Computing on device
    x/= 0;
    print('x', x);
    if (x.value() != T(8))
	throw "Error computing *= on device.";

    x+= 4;   // Computing on device
    print('x', x);
    if (x.value() != T(12))
	throw "Error computing *= on device.";

    x-= 5;   // Computing on device
    print('x', x);
    if (x.value() != T(7))
	throw "Error computing *= on device.";

    
}


int main(int argc, char* argv[])
{
    using namespace mtl;
 
    cuda::activate_best_gpu();
 //   cuda::activate_gpu(1);    
    
    std::cout<<"CUDA Scalar Test\n";
    test<int>("int");
  //  test<short>("short");
#if 0
    test<char>("char");
    test<float>("float");
    test<double>("double");
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");
#endif
    return 0;
}
