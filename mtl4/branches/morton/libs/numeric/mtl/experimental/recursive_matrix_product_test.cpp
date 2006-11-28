// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>



using namespace mtl;
using namespace std;  



// BaseCaseTest must have static information
template <typename RecuratorA, typename RecuratorB, typename RecuratorC, 
	  typename BaseCase, typename BaseCaseTest>
void recurator_mult_add(RecuratorA const& rec_a, RecuratorB const& rec_b, 
			RecuratorC& rec_c, BaseCase const& base_case, BaseCaseTest const& test)
{
    if (test(rec_a)) {
	typename RecuratorC::matrix_type c(rec_c.get_value());
	base_case(rec_a.get_value(), rec_b.get_value(), c);
	std::cout << "C after base case multiplication\n"; print_matrix_row_cursor(c);
    } else {
	RecuratorC c_north_west= rec_c.north_west(), c_north_east= rec_c.north_east(),
	           c_south_west= rec_c.south_west(), c_south_east= rec_c.south_east();

	recurator_mult_add(rec_a.north_west(), rec_b.north_west(), c_north_west, base_case, test);
	recurator_mult_add(rec_a.north_east(), rec_b.south_west(), c_north_west, base_case, test);
	recurator_mult_add(rec_a.north_west(), rec_b.north_east(), c_north_east, base_case, test);
	recurator_mult_add(rec_a.north_east(), rec_b.south_east(), c_north_east, base_case, test);
	recurator_mult_add(rec_a.south_west(), rec_b.north_west(), c_south_west, base_case, test);
	recurator_mult_add(rec_a.south_east(), rec_b.south_west(), c_south_west, base_case, test);
	recurator_mult_add(rec_a.south_west(), rec_b.north_east(), c_south_east, base_case, test);
	recurator_mult_add(rec_a.south_east(), rec_b.south_east(), c_south_east, base_case, test);
    }
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void recursive_mult_add_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    using recursion::matrix_recurator;
    matrix_recurator<MatrixA>    rec_a(a);
    matrix_recurator<MatrixB>    rec_b(b);
    matrix_recurator<MatrixC>    rec_c(c);
    equalize_depth(rec_a, rec_b, rec_c);

    // cout << "wart mal\n";
    functor::mult_add_simple_t<MatrixA, MatrixB, MatrixC> multiplicator;
    recurator_mult_add(rec_a, rec_b, rec_c, multiplicator, recursion::max_dim_test_static<4>());
    // recurator_mult_add(rec_a, rec_b, rec_c, functor::mult_add_simple_t(), recursion::max_dim_test_static<4>());
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void recursive_matrix_mult_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    recursive_mult_add_simple(a, b, c);
}









int test_main(int argc, char* argv[])
{
    //morton_dense<double,  0x55555555>      mda(3, 7), mdb(7, 2), mdc(3, 2);
    morton_dense<double,  0x55555555>      mda(5, 7), mdb(7, 6), mdc(5, 6);
    fill_hessian_matrix(mda, 1.0); fill_hessian_matrix(mdb, 2.0);
    std::cout << "mda:\n";    print_matrix_row_cursor(mda);
    std::cout << "\nmdb:\n";  print_matrix_row_cursor(mdb);

    recursive_matrix_mult_simple(mda, mdb, mdc);
    std::cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_hessian_matrix_product(mdc, 7);

    mtl::dense2D<double> da(5, 7), db(7, 6), dc(5, 6);
    fill_hessian_matrix(da, 1.0); fill_hessian_matrix(db, 2.0);
    std::cout << "\nda:\n";   print_matrix_row_cursor(da);
    std::cout << "\ndb:\n";   print_matrix_row_cursor(db);

    recursive_matrix_mult_simple(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);











    return 0;

    std::cout << "\nNow with fast pseudo dot product\n\n";

#if 0
    matrix_mult_fast_dot(mda, mdb, mdc);
    std::cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_hessian_matrix_product(mdc, 7);
#endif

    matrix_mult_fast_dot(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    
    mtl::dense2D<double> da8(8, 8), db8(8, 8), dc8(8, 8);
    fill_hessian_matrix(da8, 1.0); fill_hessian_matrix(db8, 2.0);
    std::cout << "\nda8:\n";   print_matrix_row_cursor(da8);
    std::cout << "\ndb8:\n";   print_matrix_row_cursor(db8);

    matrix_mult_fast_middle(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult_fast_middle(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    matrix_mult_fast_outer(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult_fast_outer(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    matrix_mult(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult<4>(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult<4, 4>(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    matrix_mult<4, 4, 4>(da8, db8, dc8);
    std::cout << "\ndc8:\n";   print_matrix_row_cursor(dc8);
    check_hessian_matrix_product(dc8, 8);

    return 0;
}




