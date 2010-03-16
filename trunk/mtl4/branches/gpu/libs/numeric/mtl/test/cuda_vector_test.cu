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
#include <boost/numeric/mtl/cuda/config.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>

template <typename T>
void test(const char* name)
{
    typedef mtl::cuda::vector<T>   vt;

    int gross=10;
    
    std::cout << name << "-- Vector Test\n"; 
    mtl::cuda::vector<T>  x(gross, 33), y(gross, 10, false), z(gross, 2);
    mtl::cuda::scalar<T>  c(7);
    std::cout << "Vector constructed.\n" << "x=" << x << "\n";
    std::cout << "Vector constructed.\n" << "y=" << y << "\n";

    x.to_host();
    x.to_device();

   std::cout << "\n\nx.size= " << size(x)<<"\n\n";
    
    
    x= 4.0;
    x.to_device();
    if (x[0] != T(4))
	std::cout<< "Error assign vector on device.\n";
    
    std::cout << "const x[1] == " << x[1] << "  Naechste zeile kommt X[1]=22\n";
    x[1]=22;
    x.to_device();
    std::cout<< "x=" << x << "\n";
  
    y= x;           // Copy on device
    if (y[1] != T(22))
	std::cout<< "Error copy vector on device.\n";
    std::cout<< "y[1]=" << y[1] << "\n";
    
    x.to_device();
    T c= 7;    //without testing
    x*= c;

    std::cout<<"\n\nc= "<<c<< "\nx=" << x << "\n";
    if (x[0] != T(28))
	std::cout<< "Error multipliying vector with scalar on device. x[0]="<<x[0]<<"\n";
    
    x.to_device();
    x+= 2;

    std::cout<< "x=" << x << "\n";
    if (x[0] != T(30))
	std::cout<< "Error adding vector with scalar on device. x[0]="<<x[0]<<"\n";
    
    x.to_device();
    x-= 10;

    std::cout<< "x=" << x << "\n";
    if (x[0] != T(20))
	std::cout<< "Error subtract vector with scalar on device. x[0]="<<x[0]<<"\n";
    
    x.to_device();
    x/= 10;

    std::cout<< "x=" << x << "\n";
    if (x[0] != T(2))
	std::cout<< "Error divide vector with scalar on device. x[0]="<<x[0]<<"\n";

}


int main(int argc, char* argv[])
{
    using namespace mtl;
cuda::activate_best_gpu();
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
