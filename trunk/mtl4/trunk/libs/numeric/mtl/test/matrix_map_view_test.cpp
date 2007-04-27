// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/mtl/matrix/hermitian_view.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>

using namespace mtl;
using namespace std;  

typedef complex<double> ct;

double value(double)
{
    return 7.0;
}

complex<double> value(complex<double>)
{
    return ct(7.0, 1.0);
}

// scaled value
double svalue(double)
{
    return 14.0;
}

ct svalue(ct)
{
    return ct(14.0, 2.0);
}

// conjugated value
double cvalue(double)
{
    return 7.0;
}

ct cvalue(ct)
{
    return ct(7.0, -1.0);
}

// complex scaled value
ct csvalue(double)
{
    return ct(0.0, 7.0);
}

ct csvalue(ct)
{
    return ct(-1.0, 7.0);
}


template <typename Matrix>
void test(Matrix& matrix, const char* name)
{
    set_to_zero(matrix);
    typename Matrix::value_type ref;

    {
	matrix::inserter<Matrix>  ins(matrix);
	ins(2, 3) << value(ref);
	ins(4, 3) << value(ref) + 1.0;
	ins(2, 5) << value(ref) + 2.0;
    }

    cout << "\n\n" << name << "\n";
    cout << "Original matrix:\n" << matrix << "\n";


    matrix::scaled_view<double, Matrix>  scaled_matrix(2.0, matrix);
    cout << "matrix  scaled with 2.0\n" << scaled_matrix << "\n";
    if (scaled_matrix(2, 3) != svalue(ref)) 
	throw "scaling wrong";
    
    matrix::conj_view<Matrix>  conj_matrix(matrix);
    cout << "conjugated matrix\n" << conj_matrix << "\n";
    if (conj_matrix(2, 3) != cvalue(ref)) 
	throw " wrong";

    matrix::scaled_view<ct, Matrix>  cscaled_matrix(ct(0.0, 1.0), matrix);
    cout << "matrix scaled with i (complex(0, 1))\n" << cscaled_matrix << "\n";
    if (cscaled_matrix(2, 3) != csvalue(ref)) 
	throw "complex scaling wrong";

    matrix::hermitian_view<Matrix>  hermitian_matrix(matrix);
    cout << "Hermitian matrix (conjugate transposed)\n" << hermitian_matrix << "\n";
    if (hermitian_matrix(3, 2) != cvalue(ref)) 
	throw "conjugate transposing  wrong";

#if 0    
    cout << "conjugated matrix (free function)\n" << conj(matrix) << "\n";
    if (conj(matrix)(2, 3) != cvalue(ref)) 
	throw "conjugating wrong";

    cout << "matrix  scaled with 2.0 (free function)\n" << scale(2.0, matrix) << "\n";
    if (scale(2.0, matrix)(2, 3) != svalue(ref)) 
	throw "scaling wrong";

    cout << "conjugated matrix \n" << conj(matrix) << "\n";
    if (conj(matrix)(2, 3) != cvalue(ref)) 
	throw "conjugating wrong";

    cout << "matrix scaled with i (complex(0, 1)) (free function)\n" << scale(ct(0.0, 1.0), matrix) << "\n";
    if (scale(ct(0.0, 1.0), matrix)(2, 3) != csvalue(ref)) 
	throw "complex scaling wrong";

    cout << "Hermitian  matrix (conjugate transposed) (free function)\n" << hermitian(matrix) << "\n";
    if (hermitian(matrix)(3, 2) != cvalue(ref)) 
	throw "conjugate transposing wrong";

#endif
}



int test_main(int argc, char* argv[])
{
    unsigned size= 7; 
    if (argc > 1) size= atoi(argv[1]); 

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
