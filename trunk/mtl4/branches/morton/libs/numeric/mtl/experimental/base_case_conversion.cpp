// $COPYRIGHT$

#include <iostream>
#include <string>
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

#if 0
template <typename Elt, typename Parameters, typename BaseCaseTest>
struct base_case_matrix<dense2D<Elt, Parameters>, BaseCaseTest>
{
    typename dense2D<Elt, Parameters>    type;
};
#endif


template <typename Elt, unsigned long Mask, typename Parameters, typename BaseCaseTest>
struct base_case_matrix<morton_dense<Elt, Mask, Parameters>, BaseCaseTest>
{
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


} // namespace mtl

#if 0
template <typename T>
struct meta_print
{
    meta_print(unsigned x, signed y) { dummy= x != y;}

    bool dummy;
};

template <typename Matrix>
void test2(Matrix const& matrix)
{
    cout << "in test2 (allgemein)\n";
}
#endif

template <typename Matrix>
void test()
{
    
    typedef recursion::max_dim_test_static<4> base_test_type;
#if 0
    static const unsigned long bs= base_test_type::base_case_size;
    static const unsigned long base_case_bits= meta_math::log_2<bs>::value;
    static const bool rm= is_k_power_base_case_row_major<base_case_bits, Matrix::mask>::value;
    cout << "base case size " << base_test_type::base_case_size 
	 << ", log_2 of it " << meta_math::log_2<bs>::value 
	 << ", row major " << rm
	 << "\n";
#endif

    typedef typename mtl::base_case_matrix<Matrix, base_test_type>::type base_type;
    base_type base_matrix;
    cout << typeid(base_matrix).name() << "\n";

    


#if 0
    cout << "base case size " << base_test_type::base_case_size 
	 << ", log_2 of it " << meta_math::log_2<bs>::value << "\n";
    meta_print<base_type> mp(-3, 3);
    test2(mp);
#endif
}


int test_main(int argc, char* argv[])
{
    typedef dense2D<int>                   d1t;
    typedef morton_dense<int, 0x55555553>  m1t; // col-major 4x4
    typedef morton_dense<int, 0x55555555>  m2t;
    typedef morton_dense<int, 0x5555555c>  m3t; // row-major 4x4
    typedef morton_dense<int, 0x555555f0>  m4t; // row-major 16x16

    test<d1t>();
    test<m1t>();
    test<m2t>();
    test<m3t>();
    test<m4t>();

    return 0;
} 
