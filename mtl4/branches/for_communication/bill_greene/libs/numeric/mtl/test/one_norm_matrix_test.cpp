// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>

#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/operation/one_norm.hpp>


using namespace mtl;
using namespace std;  


template <typename MatrixA>
void test(MatrixA& a, const char* name)
{
    {
	matrix::inserter<MatrixA>  ins(a);
	                  ins(0, 1) << 1.0; ins(0, 2) << 4.0;
	ins(1, 0) << 1.0; ins(1, 1) << 3.0; ins(1, 2) << 4.0; ins(1, 3) << 4.0; 
	                  ins(2, 1) << 9.0; ins(2, 2) << 4.0; ins(2, 3) << 2.0; 
	                                    ins(3, 2) << 4.0;
    }

    std::cout << "\n" << name << " a = \n" << a << "\n"
    	      << "one_norm(a) = " << one_norm(a) << "\n"; std::cout.flush();

    
    if (one_norm(a) != 16.0) throw "wrong one_norm"; 
}


int test_main(int argc, char* argv[])
{
    unsigned size= 4;

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppled_2_row_mask>  d2r(size, size);
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
 














