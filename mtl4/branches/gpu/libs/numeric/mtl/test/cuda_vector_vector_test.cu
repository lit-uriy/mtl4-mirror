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

    int size= 10270000;
    std::cout << name << "-- Vector Test\n"; 
    mtl::cuda::vector<T>  x(size, 33), y(size, 10, false), z(size, 1);
   
    y[1]= 12.0;
//    x.to_host(); y.to_host(); z.to_host();
    x.to_device(); y.to_device(); z.to_device();
    std::cout<< "X=" << x[1] << "\n";
    std::cout<< "Y=" << y[1] << "\n";
    std::cout<< "Z=" << z[1] << "\n";
    x.to_device(); y.to_device(); z.to_device();
    x= y+z;
    std::cout<< "\n>>>>>>>X= Y+ Z <<<<<<\nX=" << x[1] << "\n";
    std::cout<< "Y=" << y[1] << "\n";
    std::cout<< "Z=" << z[1] << "\n";  
    if (x[0] != T(11))
	std::cout<< "Error adding vector and vector on device.";
    x.to_device(); y.to_device(); z.to_device();
    x= y-z;    
    std::cout<< "\n>>>>>>>X= Y- Z <<<<<<\nX=" << x[1] << "\n";
    std::cout<< "Y=" << y[1] << "\n";
    std::cout<< "Z=" << z[1] << "\n";
    if (x[0] != T(9))
	std::cout<< "Error subtract vector and vector on device.";
    z*= 2;
 //we don't need this at the moment
 x.to_device(); y.to_device(); z.to_device();
    x= y*z;   
    std::cout<< "\n\n>>>>>>>X= Y* Z <<<<<<\n\nX=" << x[1] << "\n";
    std::cout<< "Y=" << y[1] << "\n";
    std::cout<< "Z=" << z[1] << "\n";   
#if 0 
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
