#include <iostream>
#include <stdio.h>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/timer.hpp>

/*
  First run: 420ns dynamic types 
             369ns static types (r6809) 
             302ns static types with unrolled by hand (r6813)
*/

#define STATIC_TYPES

using namespace mtl;

#ifdef STATIC_TYPES
   typedef dense_vector<double, vector::parameters<tag::col_major, vector::fixed::dimension<3>, true> > vec;
   typedef matrix::parameters<tag::row_major, mtl::index::c_index, mtl::fixed::dimensions<3, 3>, true> mat_para;
   #define VEC_ARG
#else
   typedef dense_vector<double> vec;
   typedef matrix::parameters<> mat_para;
   #define VEC_ARG (3)
#endif
   typedef dense2D<double, mat_para> mat;


int main()
{
  {
    vec u(3), v(3);
    u= 3., 4, 6; 

    mat A(3, 3);
    A=  2, 4, 8,
	8, 9, 1,
        4, 2, 1;

    const int rep= 10000000;
    boost::timer time;
    for(int i= 0; i < rep; i++) {
	v= A * u;
    }
    std::cout << "Compute time = " << 1000000000.*time.elapsed() / rep << "ns" << std::endl;
  }

  if (0) {
    vec u(5), v(5);
    u= 3., 4, 6, 7, 9; 

    mat A(5, 5);
    A=  2, 4, 8, 0, 8,
	8, 9, 1, 2, 1,
	8, 9, 1, 2, 1,
	8, 9, 1, 2, 1,
        4, 2, 1, 4, 9;
    std::cout << "Nach Initialisierung\n";

    const int rep= 10000000;
    boost::timer time;
    for(int i= 0; i < rep; i++) {
	v= A * u;
    }
    std::cout << "Compute time = " << 1000000000.*time.elapsed() / rep << "ns" << std::endl;
  }
  return 0;
}
