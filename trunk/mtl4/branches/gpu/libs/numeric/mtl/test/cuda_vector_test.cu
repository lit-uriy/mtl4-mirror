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
    mtl::cuda::vector<T>  x(3, 33), y(3, 10, false), z(3, 1), a(3,0);
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
    x*= 7;
    std::cout<< "x=" << x << "\n";
    x+= 2;
    std::cout<< "x=" << x << "\n";
    x-= 10;
    std::cout<< "x=" << x << "\n";
    x/= 10;
    std::cout<< "x=" << x << "\n";
    std::cout<< "y=" << y << "\n";
     x.to_device(); y.to_device(); z.to_device();
    x= y+z;
    std::cout<< "Hello\n";
    std::cout<< "x=" << x << "\n";
   // std::cout<< "y=" << y << "\n";
}


int main(int argc, char* argv[])
{
    using namespace mtl;

    test<int>("int");
//    test<short>("short");

    //test<char>("char"); // works but annoying print outs :-/
//    test<float>("float");
//    test<double>("double");

#if 0 // CUDA is too dumb for complex
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");
#endif

    return 0;
}
