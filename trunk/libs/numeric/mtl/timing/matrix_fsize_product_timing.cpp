#include <iostream>
#include <stdio.h>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/timer.hpp>

/*
  
First run:  dynamic r8530

     Compute time on 2 by 2 matrix = 415ns
     Compute time on 3 by 3 matrix = 513ns
     Compute time on 4 by 4 matrix = 476ns
     Compute time on 5 by 5 matrix = 659ns
     Compute time on 6 by 6 matrix = 793ns
     Compute time on 7 by 7 matrix = 1242ns
     Compute time on 8 by 8 matrix = 985ns

Static

     Compute time on 2 by 2 matrix = 252ns
     Compute time on 3 by 3 matrix = 270ns
     Compute time on 4 by 4 matrix = 451ns  (not inlined)
     Compute time on 5 by 5 matrix = 685ns  (not inlined)
     Compute time on 6 by 6 matrix = 987ns  (not inlined)
     Compute time on 7 by 7 matrix = 1501ns (not inlined)
     Compute time on 8 by 8 matrix = 2279ns (not inlined)


*/

using namespace mtl;

#define STATIC_TYPES

template <unsigned Size>
inline void bench()
{
    const int rep= 10000000;

#ifdef STATIC_TYPES
    typedef matrix::parameters<tag::row_major, mtl::index::c_index, mtl::fixed::dimensions<Size, Size>, true> mat_para;
#else
    typedef matrix::parameters<> mat_para;
#endif
    typedef dense2D<double, mat_para> mat;

    mat A(Size, Size);
    mtl::matrix::hessian_setup(A, 1.0);
    mat B(2*A), C(Size, Size);
    C= 0.0;

    boost::timer time;
    static const unsigned s= Size;
    asm("#before loop");
    for(int i= 0; i < rep; i++) 
	C+= A * B;
    asm("#after loop");
    
    std::cout << "Compute time on " << s << " by " << s << " matrix = " 
	      << 1000000000.*time.elapsed() / rep << "ns\n" << "C is\n" << C << std::endl;
}


int main()
{
    bench<2>();
    bench<3>();
    bench<4>();
    bench<5>();
    bench<6>();
    bench<7>();
    bench<8>();

    return 0;
}