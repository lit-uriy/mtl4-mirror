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



#define print(v) std::cout << #v << ' '; short_print2(v);
template < typename Vector>
void short_print2(const Vector& v)
{
   std::cout <<(v.valid_host()==true ? "is on Host " : "is on Device " )<< "[";
   for (int i= 0; i < 10 && i < size(v); i++)
     std::cout << v[i] << ", ";
   if(size(v)> 20) {
       std::cout << "... ,";
       for (int i= size(v)-10; i < size(v); i++)
	   std::cout << v[i] << ", ";
   }
   std::cout << "\b\b] \n";
   v.to_device();
}




template <typename T>
void test(const char* name)
{
    typedef mtl::cuda::vector<T>   vt;

    int gross=40000000;
    
    ///creating variables
    std::cout << name << "-- Vector Test\n"; 
    mtl::cuda::vector<T>  x(gross, 33), y(gross, 10, false), z(gross, 0);
    mtl::cuda::scalar<T>  c(7);

   std::cout << "\n\nVector Size= " << size(x)<<"\n\n";    
    std::cout << "Vector constructed.\n"; 
    print(x);
    std::cout << "Vector constructed.\n";
    print(y);

    x= 4.0;
    x.to_device();
    if (x[0] != T(4))
	std::cout<< "\nError assign vector on device.\n";
    
//    std::cout << "const x[1] == " << x[1] << "  Naechste zeile kommt X[1]=22\n";
    x[1]=22;
    x.to_device();
    print(x);
//  std::cout<< "x=" << x << "\n";
  
    y= x;           // Copy on device
    if (y[1] != T(22))
	std::cout<< "\nError copy vector on device.\n";
    print(y);
//    std::cout<< "y[1]=" << y[1] << "\n";
    
    x.to_device();
    x*= 7;

//    std::cout<<"\n\nc= "<<c<< "\nx=" << x << "\n";
    if (x[0] != T(28))
	std::cout<< "\nError multipliying vector with scalar on device.\n";
    
    x.to_device();
    x+= 2;

//    std::cout<< "x=" << x << "\n";    
    print(x);
    if (x[0] != T(30))
	std::cout<< "\nError adding vector with scalar on device.\n";
    
    x.to_device();
    x-= 10;

    print(x);
    std::cout<< "\n   start plus updated\n";
    x=1;
    y=1;
    print(x);
    print(y);
    x.plus_updated(y,z);
    print(z);
    unsigned elements=0;
    for(unsigned i=0; i<size(z); i++){
	if(z[i]==2) elements ++;
//	else std::cout<<"z["<<i<<"]= "<<z[i]<<"\n";
    }
    std::cout<< "   end plus updated nr elements= "<<elements<<"\n\n";
    
    
    
//    std::cout<< "x=" << x << "\n";
    print(x);
    if (x[0] != T(1))
	std::cout<< "\nError subtract vector with scalar on device.\n";
    
    x.to_device();
    x/= 10;

//    std::cout<< "x=" << x << "\n";
    print(x);    
    if (x[0] != T(0))
	std::cout<< "\nError divide vector with scalar on device.\n";

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
