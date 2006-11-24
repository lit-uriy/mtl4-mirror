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
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>

#include <boost/numeric/meta_math/log_2.hpp>
#include <boost/numeric/meta_math/is_power_of_2.hpp>

#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>

using namespace mtl;
using namespace std;  
 
namespace mtl {

template <typename Matrix, typename BaseCaseTest>
struct base_case_matrix
{
    typedef Matrix type;
};

template <typename Elt, unsigned long Mask, typename Parameters, typename BaseCaseTest>
struct base_case_matrix<morton_dense<Elt, Mask, Parameters>, BaseCaseTest>
{
    BOOST_STATIC_ASSERT(meta_math::is_power_of_2<BaseCaseTest::base_case_size>::value);
    static const unsigned long base_case_bits= meta_math::log_2<BaseCaseTest::base_case_size>::value;

    typedef typename boost::mpl::if_<
	is_k_power_base_case_row_major<base_case_bits, Mask>
      , dense2D<Elt, matrix_parameters<row_major> >
      , typename boost::mpl::if_<
	    is_k_power_base_case_col_major<base_case_bits, Mask>
	  , dense2D<Elt, matrix_parameters<col_major> >
          , morton_dense<Elt, Mask, Parameters>
        >::type
    >::type type;
};

namespace impl {

    // Without conversion, i.e. when target and source type are identical
    template <typename Matrix>
    inline Matrix 
    simplify_base_case_matrix(Matrix const& matrix, Matrix const&)
    {
	return matrix;
    }

    // With conversion, i.e. when target and source type are different
    template <typename Matrix, typename BaseCaseMatrix>
    inline BaseCaseMatrix 
    simplify_base_case_matrix(Matrix const& matrix, BaseCaseMatrix const&)
    {
	return BaseCaseMatrix(non_fixed::dimensions(matrix.num_rows(), matrix.num_cols()),
			      &const_cast<Matrix&>(matrix)[matrix.begin_row()][matrix.begin_col()]);
    }


} // namespace impl

template <typename Matrix, typename BaseCaseTest>
typename base_case_matrix<Matrix, BaseCaseTest>::type inline
simplify_base_case_matrix(Matrix const& matrix, BaseCaseTest const&)
{
    // cout << "simplify dim " <<  matrix.num_rows() << ", " << matrix.num_cols() << "\n";
    if (matrix.num_rows() != BaseCaseTest::base_case_size 
	|| matrix.num_cols() != BaseCaseTest::base_case_size) throw "Base case and matrix have different dimensions";
    return impl::simplify_base_case_matrix(matrix, base_case_matrix<Matrix, BaseCaseTest>::type());
}

} // namespace mtl






template <typename Matrix>
void test(Matrix& matrix)
{    
    typedef recursion::max_dim_test_static<4>   base_test_type;
    base_test_type                              base_test;

    typedef typename mtl::base_case_matrix<Matrix, base_test_type>::type base_type;
    base_type base_matrix;
    cout << typeid(base_matrix).name() << "\n";
    
    Matrix sm= sub_matrix(matrix, 0, 4, 0, 4);
    // cout << typeid(simplify_base_case_matrix(sm, base_test)).name() << "\n";
    typename base_case_matrix<Matrix, base_test_type>::type simplified(simplify_base_case_matrix(sm, base_test));
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

    return 0;
} 
