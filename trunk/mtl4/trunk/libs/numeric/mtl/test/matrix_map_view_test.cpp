// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/recursion/predefined_bitmask.hpp>

using namespace mtl;
using namespace std;  

template <typename Matrix>
void test(const Matrix& matrix)
{
    
    



}



int test_main(int argc, char* argv[])
{
    unsigned size= 7; 
    if (argc > 1) size= atoi(argv[1]); 

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix_parameters<col_major> >       dc(size, size);
    morton_dense<double,  recursion::morton_z_mask>      md(size, size);
    morton_dense<double,  doppler_2_row_mask>            d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix_parameters<col_major> >  cc(size, size);

    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(mzd, "Morton Z-order");
    test(d2r, "Hybrid 2 row-major");
    test(cr, "Compressed row major");
    test(cc, "Compressed column major");

    return 0;
}
