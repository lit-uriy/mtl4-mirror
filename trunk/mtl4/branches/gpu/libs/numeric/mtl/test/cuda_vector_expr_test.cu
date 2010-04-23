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
#include <boost/numeric/mtl/cuda/vector_cuda.cu>
#include <boost/numeric/mtl/cuda/dot.cu>
#include <boost/numeric/mtl/vector/all_vec_expr.hpp>

template <typename T>
void test(const char* name)
{
    typedef mtl::cuda::vector<T>   vt;

    // int critic_point=33553919, gross=53553919;
    int critic_point=5, gross=10;
    
    std::cout << name << "-- Vector Test\n"; 
    mtl::cuda::vector<T>  x(gross, 33), y(gross, 10, false), z(gross, 3);

    std::cout << "Vector Size= " << size(x) <<"\n\n"; 
   
    y[8]= 12; 
   
    std::cout<< "\n>>>>>>>Vectors Contructed <<<<<<\n";
    std::cout<< "Y= "<< y;
    std::cout<< "Z= "<< z;
    std::cout<< "X= "<< x; 
    
    x= y+z;
    std::cout<< "\n>>>>>>>X= Y + Z <<<<<<\n";
    std::cout<< "Y= "<< y;
    std::cout<< "Z= "<< z;
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(13))
	std::cout<< "Error adding vector and vector on device.";
 
    x= y-z;    
    std::cout<< "\n>>>>>>>X= Y - Z <<<<<<\n";
    std::cout<< "Y= "<< y;
    std::cout<< "Z= "<< z;
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(7))
	std::cout<< "Error subtract vector and vector on device.";

    x+= y+z;
    std::cout<< "\n>>>>>>>X+= Y + Z <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(20))
	std::cout<< "Error in x+= y+z on device.";
    
     x-= y+z;
    std::cout<< "\n>>>>>>>X-= Y + Z <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(7))
	std::cout<< "Error in x-= y+z on device.";
    
#if 0	
    x= 2 * y + z + 3 * z;
    std::cout<< "\n>>>>>>> x= 2 * y + z + 3 * z <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(32))
	std::cout<< "Error in x= 2 * y + z + 3 * z on device.";
   
    x+= y + z + dot(y, z) * z;
    std::cout<< "\n>>>>>>> x+= y + z + dot(y, z) * z <<<<<<\n";
    std::cout<< "X= "<< x;
   
    x= -z;
    std::cout<< "\n>>>>>>> x= -z <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(-3))
	std::cout<< "Error in x= -z on device.";
    
    x= y + -z;
    std::cout<< "\n>>>>>>> x= y + -z <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(7))
	std::cout<< "Error in x= y + -z on device.";
    
    x= y / 2 + z;
    std::cout<< "\n>>>>>>> x= y / 2 + z <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(8))
	std::cout<< "Error in x= y / 2 + z on device.";
    
    x= 5;
    std::cout<< "\n>>>>>>> x= 5 <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(5))
	std::cout<< "Error in x= 5 on device.";
      
    x*= 5;
    std::cout<< "\n>>>>>>> x*= 5 <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(25))
	std::cout<< "Error in x*= 5 on device.";
      
    x/= 5;
    std::cout<< "\n>>>>>>> x/= 5 <<<<<<\n";
    std::cout<< "X= "<< x;
    if (x[critic_point] != T(5))
	std::cout<< "Error in x/= 5 on device.";
#endif
}


int main(int argc, char* argv[])
{
    using namespace mtl;

    // cuda::env(argc, argv);
    cuda::activate_best_gpu();
//    cuda::activate_gpu(0);
    
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
