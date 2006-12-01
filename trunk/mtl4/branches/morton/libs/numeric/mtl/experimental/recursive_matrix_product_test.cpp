// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/timer.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>
#include <boost/numeric/mtl/recursion/recursive_matrix_mult.hpp>

using namespace mtl;
using namespace mtl::recursion;
using namespace std;  



template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA const& a, MatrixB const& b, MatrixC& c,
	  const char* name)
{
    std::cout << "\nMatrix type(s): " << name << "\n";
    //recursion::max_dim_test_static<4>    base_case_test;
    recursion::bound_test_static<4>    base_case_test;

    std::cout << "Result simple recursive multiplication:\n";
    recursive_matrix_mult_simple(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 7);

    std::cout << "Result recursive multiplication with unrolling inner loop:\n";
    recursive_matrix_mult_fast_inner(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 7);

    std::cout << "Result recursive multiplication with unrolling inner and middle loop:\n";
    recursive_matrix_mult_fast_middle<4, 8>(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 7);

    std::cout << "Result recursive multiplication with unrolling all loops:\n";
    recursive_matrix_mult_fast_outer<2, 2, 8>(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 7);
}

template <typename MatrixA, typename MatrixB, typename MatrixC>
void test_pointer(MatrixA const& a, MatrixB const& b, MatrixC& c,
		  const char* name)
{
    std::cout << "\nMatrix type(s): " << name << "\n";
    //recursion::max_dim_test_static<32>    base_case_test;
    recursion::bound_test_static<32>    base_case_test;

    std::cout << "Result recursive multiplication with pointers:\n";

    typedef functor::mult_add_row_times_col_major_32_t   fast_mult_type;
    recursive_matrix_mult<fast_mult_type, fast_mult_type>(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 32);
}




int test_main(int argc, char* argv[])
{
    // Bitmasks:

    const unsigned long morton_mask= generate_mask<true, 0, row_major, 0>::value,
	morton_z_mask= generate_mask<false, 0, row_major, 0>::value,
	doppler_4_row_mask= generate_mask<true, 2, row_major, 0>::value,
	doppler_4_col_mask= generate_mask<true, 2, col_major, 0>::value,
	doppler_16_row_mask= generate_mask<true, 4, row_major, 0>::value,
	doppler_16_col_mask= generate_mask<true, 4, col_major, 0>::value,
	doppler_32_row_mask= generate_mask<true, 5, row_major, 0>::value,
	doppler_32_col_mask= generate_mask<true, 5, col_major, 0>::value;

    morton_dense<double, morton_z_mask>      mda(5, 7), mdb(7, 9), mdc(5, 9);
    fill_hessian_matrix(mda, 1.0); fill_hessian_matrix(mdb, 2.0);

    // Hybrid col-major
    morton_dense<double, doppler_4_col_mask>   mca(5, 7), mcb(7, 9), mcc(5, 9);
    morton_dense<double, doppler_32_col_mask>  mcb32(32, 32);
    fill_hessian_matrix(mca, 1.0); fill_hessian_matrix(mcb, 2.0); fill_hessian_matrix(mcb32, 2.0);

    // Hybrid row-major
    morton_dense<double, doppler_4_row_mask>   mra(5, 7), mrb(7, 9), mrc(5, 9);
    morton_dense<double, doppler_32_row_mask>  mra32(32, 32), mrc32(32, 32);
    fill_hessian_matrix(mra, 1.0); fill_hessian_matrix(mrb, 2.0); fill_hessian_matrix(mra32, 1.0); 

    mtl::dense2D<double> da(5, 7), db(7, 9), dc(5, 9);
    fill_hessian_matrix(da, 1.0); fill_hessian_matrix(db, 2.0);

    test_pointer(mra32, mcb32, mrc32, "Hybrid col-major and row-major");
 
    test(mda, mdb, mdc, "pure Morton");
    test(da, db, dc, "dense2D");
    test(mra, mrb, mrc, "Hybrid row-major");
    test(mca, mcb, mcc, "Hybrid col-major");
    test(mra, mcb, mrc, "Hybrid col-major and row-major");
    test(mra, db, mrc, "dense2D and row-major");

    return 0;
}


