// $COPYRIGHT$

#include <iostream>
#include <string>
#include <boost/static_assert.hpp>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/print.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
//#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
//#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>

//#include <boost/numeric/meta_math/log_2.hpp>
//#include <boost/numeric/meta_math/is_power_of_2.hpp>

//#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/simplify_base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/enable_fast_dense_matrix_mult.hpp>

using namespace mtl;
using namespace std;  
 
using mtl::recursion::base_case_matrix;
using mtl::recursion::simplify_base_case_matrix;
using mtl::recursion::enable_fast_dense_matrix_mult;


template <typename Matrix>
void test(Matrix& matrix)
{    
    using mtl::recursion::base_case_matrix;

    typedef recursion::max_dim_test_static<4>   base_test_type;
    base_test_type                              base_test;

    typedef typename base_case_matrix<Matrix, base_test_type>::type base_type;
    base_type base_matrix;
    cout << typeid(base_matrix).name() << "\n";
    
    Matrix sm= sub_matrix(matrix, 0, 4, 0, 4);
    // cout << typeid(simplify_base_case_matrix(sm, base_test)).name() << "\n";
    typename base_case_matrix<Matrix, base_test_type>::type simplified(simplify_base_case_matrix(sm, base_test));
}


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void test_mult(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test,
	       const char* name)
{
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    std::cout << name << ": enable_fast_dense_matrix_mult: " 
	      << enable_fast_dense_matrix_mult<base_a_type, base_b_type, base_c_type>::value << "\n";
}





int test_main(int argc, char* argv[])
{
    typedef dense2D<int>                   d1t;
    typedef morton_dense<int, 0x55555553>  m1t; // col-major 4x4
    typedef morton_dense<int, 0x55555555>  m2t;
    typedef morton_dense<int, 0x5555555c>  m3t; // row-major 4x4
    typedef morton_dense<int, 0x555555f0>  m4t; // row-major 16x16

    d1t d1(8,8); m1t m1(8,8); m2t m2(8,8); m3t m3(8,8); m4t m4(8,8);

    test(d1);
    test(m1);
    test(m2);
    test(m3);
    test(m4);

    recursion::max_dim_test_static<4> base_test;
    test_mult(d1, d1, d1, base_test, "Only dense");
    test_mult(d1, d1, m2, base_test, "Dense and pure Morton");
    test_mult(m2, m2, m2, base_test, "Only pure Morton");
    test_mult(m1, m1, m1, base_test, "col-major hybrid");
    test_mult(m1, m1, d1, base_test, "col-major hybrid and dense");
    test_mult(m1, m1, m3, base_test, "col-major hybrid and row-major hybrid");
    test_mult(m1, m3, d1, base_test, "col-major hybrid, row-major hybrid and dense");
    test_mult(m1, m1, m2, base_test, "col-major hybrid and pure Morton");


    return 0;
} 
