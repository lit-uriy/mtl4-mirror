// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/dense2D.hpp> 
#include <boost/numeric/mtl/matrix/laplacian_setup.hpp> 
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/operation/operators.hpp>
#include <boost/numeric/mtl/operation/right_scale_inplace.hpp>

using namespace mtl;
using namespace std;  

template <typename MatrixA, typename MatrixB>
void test(MatrixA& a, MatrixB& b, unsigned dim1, unsigned dim2, const char* name)
{
    unsigned size= dim1 * dim2;
    if (size == 0)
	throw "Matrix size must be larger than 0 to make the test meaningful.";

    const unsigned max_print_size= 25;
    cout << "\n" << name << "\n";
    laplacian_setup(a, dim1, dim2);
    laplacian_setup(b, dim1, dim2);

    // right_scale_inplace(a, 2.0);
    a*= 2.0;
    if (size <= max_print_size)
	cout << "A= \n\n" << a << "\n";

    typename Collection<MatrixA>::value_type eight(8.0);
    if (a[0][0] != eight)
	throw "Scaling with scalar wrong";

    a*= 0.5; // right_scale_inplace(a, 0.5);
    a*= b;   // right_scale_inplace(a, b);

    if (size <= max_print_size)
	cout << "A= \n\n" << a << "B= \n\n" << b << "\n";

    // Check for stencil below in the middle of the matrix
    //        1
    //     2 -8  2
    //  1 -8 20 -8  1
    //     2 -8  2
    //        1    
    if (dim1 == 5 && dim2 == 5) {
	typename Collection<MatrixA>::value_type twenty(20.0), two(2.0), one(1.0), 
	                                         zero(0.0), minus_eight(-8.0);
	if (a[12][12] != twenty)
	    throw "wrong diagonal";
	if (a[12][13] != minus_eight)
	    throw "wrong east neighbor";
	if (a[12][14] != one)
	    throw "wrong east east neighbor";
	if (a[12][15] != zero)
	    throw "wrong zero-element";
	if (a[12][17] != minus_eight)
	    throw "wrong south neighbor";
	if (a[12][18] != two)
	    throw "wrong south east neighbor";
	if (a[12][22] != one)
	    throw "wrong south south neighbor";
    }
}



int test_main(int argc, char* argv[])
{
    unsigned dim1= 5, dim2= 5;

    if (argc > 2) {
	dim1= atoi(argv[1]); 
	dim2= atoi(argv[2]);
    }
    unsigned size= dim1 * dim2; 

    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);

    test(cr, dr, dim1, dim2, "Row-major sparse scaled with row-major dense");
    test(cr, dc, dim1, dim2, "Row-major sparse scaled with column-major dense");
    test(cc, dr, dim1, dim2, "Column-major sparse scaled with row-major dense");
    test(cc, dc, dim1, dim2, "Column-major sparse scaled with column-major dense");

    test(dr, cr, dim1, dim2, "Row-major dense scaled with row-major sparse");
    test(dr, cc, dim1, dim2, "Row-major dense scaled with column-major sparse");
    test(dc, cr, dim1, dim2, "Column-major dense scaled with row-major sparse");
    test(dc, cc, dim1, dim2, "Column-major dense scaled with column-major sparse");

    return 0;
}
