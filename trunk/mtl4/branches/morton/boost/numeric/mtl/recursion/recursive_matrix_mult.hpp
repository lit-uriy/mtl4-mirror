// $COPYRIGHT$

#ifndef MTL_RECURSIVE_MATRIX_MULT_INCLUDE
#define MTL_RECURSIVE_MATRIX_MULT_INCLUDE

#include <cmath>
#include <boost/mpl/if.hpp>
#include <iostream>


#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/simplify_base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/enable_fast_dense_matrix_mult.hpp>


namespace mtl { namespace recursion {

 
// BaseCaseTest must have static information
template <typename RecuratorA, typename RecuratorB, typename RecuratorC, 
	  typename BaseCase, typename BaseCaseTest>
void recurator_mult_add(RecuratorA const& rec_a, RecuratorB const& rec_b, 
			RecuratorC& rec_c, BaseCase const& base_case, BaseCaseTest const& test)
{ 
    if (rec_a.is_empty() || rec_b.is_empty() || rec_c.is_empty())
	return;

    if (test(rec_a)) { 
	typename base_case_matrix<typename RecuratorC::matrix_type, BaseCaseTest>::type
	    c(simplify_base_case_matrix(rec_c.get_value(), test)); 
	base_case(simplify_base_case_matrix(rec_a.get_value(), test), 
		  simplify_base_case_matrix(rec_b.get_value(), test), 
		  c);
#if 0
	std::cout << "A as base matrix\n"; print_matrix_row_cursor(simplify_base_case_matrix(rec_a.get_value(), test));
	std::cout << "A in original format\n"; print_matrix_row_cursor(rec_a.get_value());
	std::cout << "B as base matrix\n"; print_matrix_row_cursor(simplify_base_case_matrix(rec_b.get_value(), test));
	std::cout << "B in original format\n"; print_matrix_row_cursor(rec_b.get_value());
	std::cout << "C as base matrix\n"; print_matrix_row_cursor(c);
	std::cout << "C in original format\n"; print_matrix_row_cursor(rec_c.get_value());
#endif
    } else {
	RecuratorC c_north_west= rec_c.north_west(), c_north_east= rec_c.north_east(),
	           c_south_west= rec_c.south_west(), c_south_east= rec_c.south_east();

	recurator_mult_add(rec_a.north_west(), rec_b.north_west(), c_north_west, base_case, test);
	recurator_mult_add(rec_a.north_west(), rec_b.north_east(), c_north_east, base_case, test);
	recurator_mult_add(rec_a.south_west(), rec_b.north_east(), c_south_east, base_case, test);
	recurator_mult_add(rec_a.south_west(), rec_b.north_west(), c_south_west, base_case, test);
	recurator_mult_add(rec_a.south_east(), rec_b.south_west(), c_south_west, base_case, test);
	recurator_mult_add(rec_a.south_east(), rec_b.south_east(), c_south_east, base_case, test);
	recurator_mult_add(rec_a.north_east(), rec_b.south_east(), c_north_east, base_case, test);
	recurator_mult_add(rec_a.north_east(), rec_b.south_west(), c_north_west, base_case, test);
    }
}


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_mult_add_simple(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    using recursion::matrix_recurator;
    matrix_recurator<MatrixA>    rec_a(a);
    matrix_recurator<MatrixB>    rec_b(b);
    matrix_recurator<MatrixC>    rec_c(c);
    equalize_depth(rec_a, rec_b, rec_c);

    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>       mult_type;

    recurator_mult_add(rec_a, rec_b, rec_c, mult_type(), test);
}


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest> 
void recursive_matrix_mult_simple(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    set_to_0(c);
    recursive_mult_add_simple(a, b, c, test);
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

    // std::std::cout << "Multiplier: " << typeid(bc).name() << "\n";
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

    typedef functor::mult_add_fast_middle_t<base_a_type, base_b_type, base_c_type, 
	                                    InnerUnroll, MiddleUnroll>              fast_mult_type;
    typedef functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>       slow_mult_type;

    recursive_matrix_mult<fast_mult_type, slow_mult_type>(a, b, c, test);
} 


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_matrix_mult_fast_middle(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    recursive_matrix_mult_fast_middle<MTL_MATRIX_MULT_INNER_UNROLL, MTL_MATRIX_MULT_MIDDLE_UNROLL>(a, b, c, test);
}


template <unsigned InnerUnroll, unsigned MiddleUnroll, unsigned OuterUnroll,
	  typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_matrix_mult_fast_outer(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef functor::mult_add_fast_outer_t<base_a_type, base_b_type, base_c_type, 
	                                   InnerUnroll, MiddleUnroll, OuterUnroll>  fast_mult_type;
    typedef functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>       slow_mult_type;

    recursive_matrix_mult<fast_mult_type, slow_mult_type>(a, b, c, test);
} 


template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest>
void recursive_matrix_mult_fast_outer(MatrixA const& a, MatrixB const& b, MatrixC& c, BaseCaseTest const& test)
{
    recursive_matrix_mult_fast_outer<MTL_MATRIX_MULT_INNER_UNROLL, MTL_MATRIX_MULT_MIDDLE_UNROLL,
	                             MTL_MATRIX_MULT_OUTER_UNROLL>(a, b, c, test);
}


}} // namespace mtl::recursion

#endif // MTL_RECURSIVE_MATRIX_MULT_INCLUDE
