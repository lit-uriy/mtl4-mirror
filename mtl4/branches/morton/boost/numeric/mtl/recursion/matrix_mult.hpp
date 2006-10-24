// $COPYRIGHT$

#ifndef MTL_RECURSION_MATRIX_MULT_INCLUDE
#define MTL_RECURSION_MATRIX_MULT_INCLUDE

#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>


namespace mtl { namespace recursion {

template <typename RecuratorA, typename RecuratorB, typename RecuratorC, typename BaseMult, typename BaseTest>
void recurator_matrix_mult(RecuratorA const& ra, RecuratorB const& rb, RecuratorC& c, 
                           BaseMult const& base_mult, BaseTest const& base_test)
{
    // Check for empty matrices
    if (ra.get_value().num_rows() == 0 || ra.get_value().num_cols() == 0 || rb.get_value().num_cols() == 0)
	return;
		
    if (base_test(ra)) { 
	base_mult(ra.get_value(), rb.get_value(), rc.get_value());
	return;
    }
		
    RecuratorC rc_nw(rc.north_west()), rc_sw(rc.south_west()), c_ne(rc.north_east()), rc_se(rc.south_east());
    
    recurator_matrix_mult(ra.north_west(), rb.north_west(), rc_nw, base_mult, base_test);
    recurator_matrix_mult(ra.north_east(), rb.south_west(), rc_nw, base_mult, base_test);
    recurator_matrix_mult(ra.north_west(), rb.north_east(), rc_ne, base_mult, base_test);
    recurator_matrix_mult(ra.north_east(), rb.south_east(), rc_ne, base_mult, base_test);
    recurator_matrix_mult(ra.south_west(), rb.north_west(), rc_sw, base_mult, base_test);
    recurator_matrix_mult(ra.south_east(), rb.south_west(), rc_sw, base_mult, base_test);
    recurator_matrix_mult(ra.south_west(), rb.north_east(), rc_se, base_mult, base_test);
    recurator_matrix_mult(ra.south_east(), rb.south_east(), rc_se, base_mult, base_test);
}


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseTest= mtl::recursion::max_dim_test_32>
void recursive_matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseTest const& base_test= BaseTest(32))
{
    using mtl::recursion::recurator;
    
    if (a.num_rows() != c.num_rows()) throw "Incompatible Matrix Sizes\n";
    if (a.num_cols() != b.num_rows()) throw "Incompatible Matrix Sizes\n";
    if (b.num_cols() != c.num_cols()) throw "Incompatible Matrix Sizes\n";
			
    // Set C to 0
			
    matrix_recurator<MatrixA>  ra(a);
    matrix_recurator<MatrixB>  rb(b);
    matrix_recurator<MatrixC>  rc(c);
    // compatible recurators to each other
    // equalize_depth(ra, rb, rc);

    mtl::recursion::base_mult<MatrixA, MatrixB, MatrixC, BaseTest> bm;

    recurator_matrix_mult(ra, rb, rc, bm, base_test);
}

}} // namespace mtl::recursion

#endif // MTL_RECURSION_MATRIX_MULT_INCLUDE
