// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/inserter.hpp> 
#include <boost/numeric/mtl/matrix/element_matrix.hpp> 
#include <boost/numeric/mtl/matrix/element_array.hpp> 
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/operation/print.hpp>

using namespace mtl;
using namespace std;  

template <typename Matrix>
void test(Matrix& matrix, const char* name)
{
    cout << "\n" << name << "\n";

    using matrix::inserter;
    typedef typename Collection<Matrix>::value_type value_type;

    dense2D<double>       m1(2, 2);
    m1[0][0]= 1.0; m1[0][1]= 2.0; 
    m1[1][0]= 3.0; m1[1][1]= 4.0; 
    std::vector<int>           row1, col1;
    row1.push_back(1); row1.push_back(2);
    col1.push_back(0); col1.push_back(2);
    

    double a2[2][2]= {{11., 12.},{13., 14.}};
    std::vector<int>           ind2;
    ind2.push_back(2); ind2.push_back(4);

    std::vector<int>           ind3;
    ind3.push_back(3); ind3.push_back(1);

    set_to_zero(matrix); // dense matrices are not automatically set to zero

    {
	inserter<Matrix, operations::update_plus<value_type> > ins(matrix);

	ins << element_matrix(m1, row1, col1)
	    << element_array(a2, ind2);
	ins << element_array(a2, ind3);
    }

    cout << "Filled matrix:\n" << matrix << "\n";
    if (matrix[0][0] != 0.0)
	throw "wrong zero-element";
    if (matrix[1][0] != 1.0)
	throw "wrong insertion (single value)";
    if (matrix[2][2] != 15.0)
	throw "wrong summation";
    if (matrix[1][1] != 14.0)
	throw "wrong insertion (single value)";
    
}



int test_main(int argc, char* argv[])
{
    unsigned size= 5;

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
