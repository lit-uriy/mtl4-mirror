// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/diagonal_setup.hpp> 
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/operation/print.hpp>


using namespace mtl;
using namespace std;  

template <typename Matrix>
void test(Matrix& matrix, const char* name)
{
    cout << "\n" << name << "\n";
    typename Collection<Matrix>::value_type four(4.0), one(1.0), zero(0.0);

    matrix::diagonal_setup(matrix, 1.0);
    if (matrix[0][0] != one)
	throw "wrong diagonal";
    if (matrix[0][1] != zero)
	throw "wrong off-diagonal";

    cout << "Diagonal matrix:\n" << matrix << "\n";

    matrix= 4.0;
    if (matrix[0][0] != four)
	throw "wrong diagonal";
    if (matrix[0][1] != zero)
	throw "wrong off-diagonal";

    cout << "Diagonal matrix:\n" << matrix << "\n";
}



int test_main(int argc, char* argv[])
{
    unsigned size= 7; 

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppler_2_row_mask>  d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);

    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(mzd, "Morton Z-order");
    test(d2r, "Hybrid 2 row-major");
    test(cr, "Compressed row major");
    test(cc, "Compressed column major");
    test(drc, "Dense row major complex");
    test(crc, "Compressed row major complex");

    return 0;
}
