// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/mpl/if.hpp>

// #include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/simplify_base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/enable_fast_dense_matrix_mult.hpp>



using namespace mtl;
using namespace mtl::recursion;
using namespace std;  



// BaseCaseTest must have static information
template <typename RecuratorA, typename RecuratorB, typename RecuratorC, 
	  typename BaseCase, typename BaseCaseTest>
void recurator_mult_add(RecuratorA const& rec_a, RecuratorB const& rec_b, 
			RecuratorC& rec_c, BaseCase const& base_case, BaseCaseTest const& test)
{ 
    if (test(rec_a)) { 
	typename base_case_matrix<typename RecuratorC::matrix_type, BaseCaseTest>::type
	    c(simplify_base_case_matrix(rec_c.get_value(), test)); 
	base_case(simplify_base_case_matrix(rec_a.get_value(), test), 
		  simplify_base_case_matrix(rec_b.get_value(), test), 
		  c);
#if 0
	cout << "A as base matrix\n"; print_matrix_row_cursor(simplify_base_case_matrix(rec_a.get_value(), test));
	cout << "A in original format\n"; print_matrix_row_cursor(rec_a.get_value());
	cout << "B as base matrix\n"; print_matrix_row_cursor(simplify_base_case_matrix(rec_b.get_value(), test));
	cout << "B in original format\n"; print_matrix_row_cursor(rec_b.get_value());
	cout << "C as base matrix\n"; print_matrix_row_cursor(c);
	cout << "C in original format\n"; print_matrix_row_cursor(rec_c.get_value());
#endif
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
    typedef recursion::max_dim_test_static<4>                      BaseCaseTest;
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>       mult_type;

    functor::mult_add_simple_t<MatrixA, MatrixB, MatrixC> multiplicator;
    recurator_mult_add(rec_a, rec_b, rec_c, mult_type(), BaseCaseTest());
    // recurator_mult_add(rec_a, rec_b, rec_c, functor::mult_add_simple_t(), recursion::max_dim_test_static<4>());
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void recursive_matrix_mult_simple(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    recursive_mult_add_simple(a, b, c);
}


template <typename FastBaseCase, typename SlowBaseCase, typename BaseCaseTest,
	  typename MatrixA, typename MatrixB, typename MatrixC>
struct base_case_dispatcher
{
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef typename boost::mpl::if_<
	recursion::enable_fast_dense_matrix_mult<base_a_type, base_b_type, base_c_type>
      , FastBaseCase
      , SlowBaseCase
    >::type base_case_type;

    void operator() (MatrixA const& a, MatrixB const& b, MatrixC& c) const
    {
	base_case_type()(a, b, c);
    }
};


template <typename FastBaseCase, typename SlowBaseCase, typename BaseCaseTest,
	  typename MatrixA, typename MatrixB, typename MatrixC>
void recursive_mult_add(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    using recursion::matrix_recurator;
    matrix_recurator<MatrixA>    rec_a(a);
    matrix_recurator<MatrixB>    rec_b(b);
    matrix_recurator<MatrixC>    rec_c(c);
    equalize_depth(rec_a, rec_b, rec_c);

    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;
    base_case_dispatcher<FastBaseCase, SlowBaseCase, BaseCaseTest, base_a_type, base_b_type, base_c_type> multiplier;
    // typename base_case_dispatcher<FastBaseCase, SlowBaseCase, BaseCaseTest, MatrixA, MatrixB, MatrixC>::base_case_type bc;

    // std::cout << "Multiplier: " << typeid(bc).name() << "\n";
    recurator_mult_add(rec_a, rec_b, rec_c, multiplier, test);
}


template <typename FastBaseCase, typename SlowBaseCase, typename BaseCaseTest,
	  typename MatrixA, typename MatrixB, typename MatrixC>
void recursive_matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    set_to_0(c);
    recursive_mult_add<FastBaseCase, SlowBaseCase>(a, b, c, test);
}


template <unsigned InnerUnroll, typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_matrix_mult_fast_inner(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef functor::mult_add_fast_inner_t<base_a_type, base_b_type, base_c_type, InnerUnroll>   fast_mult_type;
    typedef functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>       slow_mult_type;

    recursive_matrix_mult<fast_mult_type, slow_mult_type>(a, b, c, test);
}


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_matrix_mult_fast_inner(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    recursive_matrix_mult_fast_inner<MTL_MATRIX_MULT_INNER_UNROLL>(a, b, c, test);
}


template <unsigned InnerUnroll, unsigned MiddleUnroll,
	  typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_matrix_mult_fast_middle(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef functor::mult_add_fast_middle_t<base_a_type, base_b_type, base_c_type, InnerUnroll>   fast_mult_type;
    typedef functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>       slow_mult_type;

    recursive_matrix_mult<fast_mult_type, slow_mult_type>(a, b, c, test);
} 


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_matrix_mult_fast_middle(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    recursive_matrix_mult_fast_middle<MTL_MATRIX_MULT_INNER_UNROLL, MTL_MATRIX_MULT_MIDDLE_UNROLL>(a, b, c, test);
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA const& a, MatrixB const& b, MatrixC& c,
	  const char* name)
{
    std::cout << "\nMatrix type(s): " << name << "\n";
    std::cout << "Result simple recursive multiplication:\n";
    recursive_matrix_mult_simple(a, b, c);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 7);

    recursion::max_dim_test_static<4>    base_case_test;

    std::cout << "Result recursive multiplication with unrolling inner loop:\n";
    recursive_matrix_mult_fast_inner(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 7);

    std::cout << "Result recursive multiplication with unrolling inner and middle loop:\n";
    recursive_matrix_mult_fast_middle(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 7);
}


int test_main(int argc, char* argv[])
{
    morton_dense<double,  0x55555555>      mda(5, 7), mdb(7, 6), mdc(5, 6);
    fill_hessian_matrix(mda, 1.0); fill_hessian_matrix(mdb, 2.0);

    // Hybrid col-major
    morton_dense<double,  0x55555553>      mca(5, 7), mcb(7, 6), mcc(5, 6);
    fill_hessian_matrix(mca, 1.0); fill_hessian_matrix(mcb, 2.0);

    // Hybrid row-major
    morton_dense<double,  0x5555555c>      mra(5, 7), mrb(7, 6), mrc(5, 6);
    fill_hessian_matrix(mra, 1.0); fill_hessian_matrix(mrb, 2.0);

    mtl::dense2D<double> da(5, 7), db(7, 6), dc(5, 6);
    fill_hessian_matrix(da, 1.0); fill_hessian_matrix(db, 2.0);

    test(mda, mdb, mdc, "pure Morton");
    test(da, db, dc, "dense2D");
    test(mra, mrb, mrc, "Hybrid row-major");
    test(mca, mcb, mcc, "Hybrid col-major");
    test(mra, mcb, mrc, "Hybrid col-major and row-major");
    test(mra, db, mrc, "dense2D and row-major");

    return 0;
}




