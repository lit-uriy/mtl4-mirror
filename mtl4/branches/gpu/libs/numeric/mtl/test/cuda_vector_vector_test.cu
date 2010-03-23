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


/* print function*/
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

    int critic_point=33553919, gross= 53553919;//critic_point+1;
    
    std::cout << name << "-- Vector Test\n"; 
    mtl::cuda::vector<T>  x(gross, 33), y(gross, 10, false), z(gross, 3);

    std::cout << "Vector Size= " << size(x) <<"\n\n";
    
    y[8]= 12;
//    x.to_host(); y.to_host(); z.to_host();
    x.to_device(); y.to_device(); z.to_device();
    
    std::cout<< "\n>>>>>>>Vectors Contructed <<<<<<\n";
    print(y);
    print(z);
    print(x);
    
    
    x= y+z;
    std::cout<< "\n>>>>>>>X= Y+ Z <<<<<<\n";
    print(y);
    print(z);
    print(x);
    if (x[critic_point] != T(13))
	std::cout<< "Error adding vector and vector on device.";
    
    x= y-z;    
    std::cout<< "\n>>>>>>>X= Y- Z <<<<<<\n";
    print(y);
    print(z);
    print(x);
    if (x[critic_point] != T(7))
	std::cout<< "Error subtract vector and vector on device.";

    //z*= 2;
    x= y*z;   
    std::cout<< "\n\n>>>>>>>X= Y* Z <<<<<<\n";
    print(y);
    print(z);
    print(x);
    if (x[critic_point] != T(30))
	std::cout<< "Error subtract vector and vector on device.";

 
/*    
    
    x=0;
    std::cout<< "\n   start plus updated\n";
    print(y);
    print(z);
    z.plus_updated(y,x);
    print(x);
    unsigned elements=0;
    for(unsigned i=0; i<size(x); i++){
	if(x[i]!=0) elements ++;
//	else std::cout<< "x["<<i<<"]= "<<x[i]<<"\n";
    }
    std::cout<< "   end plus updated nr elements of vector x= "<<elements<<"\n\nx[1]="<<x[1]<<"\n\n";
    */
    
    
    
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

    cuda::activate_best_gpu();
    
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
