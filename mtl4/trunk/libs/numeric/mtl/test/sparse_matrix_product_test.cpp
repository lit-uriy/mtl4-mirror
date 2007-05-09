// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/laplacian_setup.hpp> 
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/operation/operators.hpp>

using namespace mtl;
using namespace std;  

template <typename MatrixA, typename MatrixB>
void test(MatrixA& a, MatrixB&, unsigned dim1, unsigned dim2, const char* name)
{
    cout << "\n" << name << "\n";
    laplacian_setup(a, dim1, dim2);
    MatrixB b;
    laplacian_setup(b, dim1, dim2);

    unsigned size= dim1 * dim2;
    compressed2D<double>  c(size, size);
    c= a * b;

    if (size <= 25)
	cout << "A= \n\n" << a << "B= \n\n" << b << "A*B= \n\n" << c << "\n";

    // Check for stencil below in the middle of the matrix
    //        1
    //     2 -8  2
    //  1 -8 20 -8  1
    //     2 -8  2
    //        1    
    if (dim1 == 5 && dim2 == 5) {
	typename Collection<MatrixA>::value_type twenty(20.0), two(2.0), one(1.0), 
	                                         zero(0.0), minus_eight(-8.0);
	if (c[12][12] != twenty)
	    throw "wrong diagonal";
	if (c[12][13] != minus_eight)
	    throw "wrong east neighbor";
	if (c[12][14] != one)
	    throw "wrong east east neighbor";
	if (c[12][15] != zero)
	    throw "wrong zero-element";
	if (c[12][17] != minus_eight)
	    throw "wrong south neighbor";
	if (c[12][18] != two)
	    throw "wrong south east neighbor";
	if (c[12][22] != one)
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

    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);

    test(cr, cr, dim1, dim2, "Row-major times row-major");
#if 0
    test(cr, cc, dim1, dim2, "Row-major times column-major");

    test(cc, cr, dim1, dim2, "Column-major times row-major");
    test(cc, cc, dim1, dim2, "Column-major times column-major");
#endif

    return 0;
}
