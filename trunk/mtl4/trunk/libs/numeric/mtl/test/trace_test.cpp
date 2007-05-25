// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/transposed_view.hpp>
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/laplacian_setup.hpp>

#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/operation/trace.hpp>


using namespace mtl;
using namespace std;  


template <typename MatrixA>
void test(MatrixA& a, unsigned dim1, unsigned dim2, const char* name)
{
    matrix::laplacian_setup(a, dim1, dim2);

    std::cout << "\n" << name << " a = \n" << a << "\n"
	      << "trace(a) = " << trace(a) << "\n"; std::cout.flush();

    // Due to rounding errors, dimensions shouldn't be too large (or test less naive)
    if (trace(a) != pow(4.0, int(dim1*dim2))) throw "wrong trace"; 
}


int test_main(int argc, char* argv[])
{
    unsigned dim1= 3, dim2= 2;

    if (argc > 2) {
	dim1= atoi(argv[1]); 
	dim2= atoi(argv[2]);
    }
    unsigned size= dim1 * dim2; 

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppler_2_row_mask>  d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);

    test(dr, dim1, dim2, "Dense row major");
    test(dc, dim1, dim2, "Dense column major");
    test(mzd, dim1, dim2, "Morton Z-order");
    test(d2r, dim1, dim2, "Hybrid 2 row-major");
    test(cr, dim1, dim2, "Compressed row major");
    test(cc, dim1, dim2, "Compressed column major");
    test(drc, dim1, dim2, "Dense row major complex");
    test(crc, dim1, dim2, "Compressed row major complex");

    return 0;
}
 














