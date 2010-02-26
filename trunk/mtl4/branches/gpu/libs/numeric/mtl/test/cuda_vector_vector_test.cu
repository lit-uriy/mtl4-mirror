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

#include <boost/numeric/mtl/cuda/vector_cuda.cu>

template <typename T>
void test(const char* name)
{
    typedef mtl::cuda::vector<T>   vt;

    std::cout << name << "-- Vector Test\n"; 
    mtl::cuda::vector<T>  x(3, 33), y(3, 10, false), z(3, 1);
   
    y[1]= 12.0;
//    x.to_host(); y.to_host(); z.to_host();
    x.to_device(); y.to_device(); z.to_device();
    std::cout<< "X=" << x << "\n";
    std::cout<< "Y=" << y << "\n";
    std::cout<< "Z=" << z << "\n";
    
    x= y+z;
    std::cout<< "\n>>>>>>>X= Y+ Z <<<<<<\nX=" << x << "\n";
    std::cout<< "Y=" << y << "\n";
    std::cout<< "Z=" << z << "\n";  
    if (x[0] != T(11))
	std::cout<< "Error adding vector and vector on device.";
    
    x= y-z;    
    std::cout<< "\n>>>>>>>X= Y- Z <<<<<<\nX=" << x << "\n";
    std::cout<< "Y=" << y << "\n";
    std::cout<< "Z=" << z << "\n";
    if (x[0] != T(9))
	std::cout<< "Error subtract vector and vector on device.";
    
#if 0  //we don't need this at the moment
    x= y*z;   
    std::cout<< "\n\n>>>>>>>X= Y* Z <<<<<<\n\nX=" << x << "\n";
    std::cout<< "Y=" << y << "\n";
    std::cout<< "Z=" << z << "\n";   
    x= y/z;
    std::cout<< "\n\n>>>>>>>X= Y/ Z <<<<<<\n\nX=" << x << "\n";
    std::cout<< "Y=" << y << "\n";
    std::cout<< "Z=" << z << "\n";   
#endif 

}


int main(int argc, char* argv[])
{
    using namespace mtl;

    test<int>("int");
  //  test<short>("short");

    //test<char>("char"); // works but annoying print outs :-/
    //test<float>("float");
    //test<double>("double");

#if 0 // CUDA is too dumb for complex
    test<std::complex<float> >("std::complex<float>");
    test<std::complex<double> >("std::complex<double>");
#endif

    return 0;
}
