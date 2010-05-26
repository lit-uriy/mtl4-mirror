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
#include <cstdlib>

#include <boost/numeric/mtl/cuda/vector_cuda.cu>
#include <boost/numeric/mtl/cuda/timer.cu>

template < typename Vector>
void short_print2(const Vector& v)
{
   std::cout << "[";
   for (int i= 0; i < 10 && i < size(v); i++)
     std::cout << v[i] << ", ";
   if(size(v)> 20) {
       std::cout << " ... ";
       for (int i= size(v)-10; i < size(v); i++)
	   std::cout << v[i] << ", ";
   }
   std::cout << "\b\b] \n";
   v.to_device();
}
  
#define short_print(v) std::cout << #v << ' '; short_print2(v);
  

template <typename T>
void test(const char* name, int size, int repetitions)
{
    typedef mtl::cuda::vector<T>   vt;

    std::cout << name << "-- Vector benchmark Test (must be reviewed because expresion templates\n"; 
    mtl::cuda::vector<T>  x(size, 33), y(size, 10, false), z(size, 1), q(size, 3);
    
#if 0
    x.plus(y, z);
    z.to_host();
    short_print(z);
#endif 	
    
    for (int r= 0; r < 10; r++)
	z= x + y;
    
    mtl::cuda::timer with_copy;
    for (int r= 0; r < repetitions; r++)
	z= x + y;
    
    double tc= 1000 * with_copy.elapsed() / float(repetitions);
    std::cout << "Time with copy " << tc << "ms, corresponds to " << size / 1000000.0 / tc << "GFLOPs (GIPs)\n";
        
    mtl::cuda::timer without_copy;
    for (int r= 0; r < repetitions; r++) {
//	x.plus(y, z);
 	cudaThreadSynchronize();
    }
   
    double tw= 1000 * without_copy.elapsed() / float(repetitions);
    std::cout << "Time without copy " << tw << "ms, corresponds to " << size / 1000000.0 / tw << "GFLOPs (GIPs)\n";

    mtl::cuda::timer twice;
    for (int r= 0; r < repetitions; r++) {
//	x.plus(y, z);           //that
//	y.plus(z, q);           //and that 
 	cudaThreadSynchronize();
    }
   
    double twi= 1000 * twice.elapsed() / float(repetitions);
    std::cout << "Time with two operations " << twi << "ms, corresponds to " << 2*size / 1000000.0 / twi << "GFLOPs (GIPs)\n";
    
#if 0
    short_print(x);
    short_print(y);
    short_print(z);
    short_print(q);
#endif
}


int main(int argc, char* argv[])
{
    using namespace mtl;

    int size= 1027000, repetitions= 200;
    if (argc > 1)
	size= atoi(argv[1]);
    if (argc > 2)
	repetitions= atoi(argv[2]);
    
    test<int>("int", size, repetitions);
    test<float>("float", size, repetitions);
    test<double>("double", size, repetitions);

    return 0;
}
