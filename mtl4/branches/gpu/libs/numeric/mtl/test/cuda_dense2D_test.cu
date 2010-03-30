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

#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>



///print matrix short
#define printm(m) std::cout<<"\n\n"<< #m << ' '; short_printm(m);
template < typename Matrix>
void short_printm(const Matrix& m)
{
   if (num_rows(m)< 30 && num_cols(m)<30) std::cout<<m<< "\n";
   else {
        std::cout <<(m.valid_host()==true ? "is on Host " : "is on Device " )<< "\n";
        for(int i=0; i<15; i++) {
	    std::cout<<"[";    
	    for(int j=0; j<15; j++) std::cout<<m(i,j)<< " ";
	    std::cout<<". . . . . ";
	    for(int j=num_cols(m)-15; j<num_cols(m); j++) std::cout<<m(i,j)<< " ";
	    std::cout<<"]\n";
       }
        
        
	for(int i=0; i<5; i++) {
	    std::cout<<"[";
	    for(int j=0; j<35; j++) std::cout<<". ";
	    std::cout<<"]\n";
       }
	
        
	for(int i=num_rows(m)-15; i<num_rows(m); i++) {
	    std::cout<<"[";
	    for(int j=0; j<15; j++) std::cout<<m(i,j)<< " ";
	    std::cout<< ". . . . . ";
	    for(int j=num_cols(m)-15; j<num_cols(m); j++) std::cout<<m(i,j)<< " ";
	    std::cout<<"]\n";
       }	
	
	   

   }
}



template <typename T>
void test(const char* name)
{
    typedef mtl::cuda::dense2D<T>   dense;
    int size= 32531;  //32531 =  limit
    std::cout << name << "-- Matrix Test\n"; 
    mtl::cuda::dense2D<T>  A(size, size);
    mtl::cuda::vector<T>  x(size, 1.0), b(size, 0.0); 
 
    std::cout << "Matrix constructed.\n";
    std::cout << "Matrix Dimension. "<<size<<"x"<<size<<"\n";
    std::cout << "Matrix Elements= "<<size*size<<"\n";

    A.set_to_zero();
    std::cout << "Matrix on host= "<< (A.valid_host()==true ?  "Yes\n\n" : "No\n\n");
    

    for (int i= 0; i < size; i++)
      A(2,i,i);
    
    A(2,0,size-1);
    A(2,size-1,0);
    
 //   A.to_device();

///Start Vector = Matrix x Vector
    std::cout<< "start multiplication\n";
    b= A * x;
    std::cout<< "ende multiplication\n";
    std::cout<< "x[7]=" << x[7] << "\n";
    std::cout<< "b[0]=" << b[0] << "\n";
    std::cout<< "b[1]=" << b[1] << "\n";

      if (b[0] != T(4) || b[1]!=T(2))
 	std::cout<< "Error Matrix vector multiplication on device.\n";
      else
	std::cout<< "Multiplication Without problems.\n";  
      std::cout<< "A= "<<A;
      //printm(A);
///End Vector = Matrix x Vector
      
      
//    A.set_to_zero(true);   //  on host
//    A.laplacian_setup_host(7);

//    A.set_to_zero();       // on device
//    A.laplacian_setup_device(5);

//     std::cout<< "\n\nA=" << A<<"\n";
     std::cout<< "A= "<<A;
     //printm(A);
   


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
