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
#include <boost/timer.hpp>



using namespace mtl;
using namespace mtl::recursion;
using namespace std;  



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




template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA const& a, MatrixB const& b, MatrixC& c,
	  const char* name)
{
    std::cout << "\nMatrix type(s): " << name << "\n";
    recursion::max_dim_test_static<4>    base_case_test;
    //recursion::bound_test_static<4>    base_case_test;

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
    recursion::max_dim_test_static<32>    base_case_test;
    //recursion::bound_test_static<32>    base_case_test;

    std::cout << "Result recursive multiplication with pointers:\n";

    typedef functor::mult_add_row_times_col_major_32_t   fast_mult_type;
    recursive_matrix_mult<fast_mult_type, fast_mult_type>(a, b, c, base_case_test);
    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, 32);
}


void print_time_and_mflops(double time, double size)
{
    std::cout << "    takes " << time << "s = " << 2.0 * size * size * size / time / 1e6f << "MFlops\n";
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void measure_mult(MatrixA const& a, MatrixB const& b, MatrixC& c,
		 const char* name)
{
    std::cout << "\nMatrix type(s): " << name << "\n";

    recursion::max_dim_test_static<32>    base_case_test;
    //recursion::bound_test_static<32>    base_case_test;

    std::cout << "Simple recursive multiplication:\n";
    boost::timer start1;
    recursive_matrix_mult_simple(a, b, c, base_case_test); 
    print_time_and_mflops(start1.elapsed(), a.num_rows());
    // std::cout << "    takes " << start1.elapsed() << "s\n";
    // print_matrix_row_cursor(c); 

    std::cout << "Recursive multiplication with unrolling inner loop:\n"; 
    boost::timer start2;
    recursive_matrix_mult_fast_inner(a, b, c, base_case_test);
    print_time_and_mflops(start2.elapsed(), a.num_rows());
    // std::cout << "    takes " << start2.elapsed() << "s\n";
    // print_matrix_row_cursor(c);

    std::cout << "Recursive multiplication with unrolling inner and middle loop:\n";
    boost::timer start3;
    recursive_matrix_mult_fast_middle<4, 8>(a, b, c, base_case_test);
    print_time_and_mflops(start3.elapsed(), a.num_rows());
    // std::cout << "    takes " << start3.elapsed() << "s\n";
    // print_matrix_row_cursor(c);

    std::cout << "Recursive multiplication with unrolling all loops:\n";
    boost::timer start4;
    recursive_matrix_mult_fast_outer<2, 2, 8>(a, b, c, base_case_test);
    print_time_and_mflops(start4.elapsed(), a.num_rows());
    // std::cout << "    takes " << start4.elapsed() << "s\n";
    // print_matrix_row_cursor(c);  
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void measure_mult_pointer(MatrixA const& a, MatrixB const& b, MatrixC& c,
				  const char* name)
{
    std::cout << "\nMult with low abstraction, Matrix type(s): " << name << "\n";

    recursion::max_dim_test_static<32>                   base_case_test;
    typedef functor::mult_add_row_times_col_major_32_t   fast_mult_type;

    boost::timer start1;
    recursive_matrix_mult<fast_mult_type, fast_mult_type>(a, b, c, base_case_test);
    print_time_and_mflops(start1.elapsed(), a.num_rows());
}



int test_main(int argc, char* argv[])
{
    // Bitmasks:

    const unsigned long morton_mask= generate_mask<true, 0, row_major, 0>::value,
	morton_z_mask= generate_mask<false, 0, row_major, 0>::value,
	doppler_4_row_mask= generate_mask<true, 2, row_major, 0>::value,
	doppler_4_col_mask= generate_mask<true, 2, col_major, 0>::value,
	doppler_32_row_mask= generate_mask<true, 5, row_major, 0>::value,
	doppler_32_col_mask= generate_mask<true, 5, col_major, 0>::value;

    // For testing:
    // ============
#if 0
    morton_dense<double,  0x55555555>      mda(5, 7), mdb(7, 6), mdc(5, 6);
    fill_hessian_matrix(mda, 1.0); fill_hessian_matrix(mdb, 2.0);

    // Hybrid col-major
    morton_dense<double,  0x55555553>      mca(5, 7), mcb(7, 6), mcc(5, 6);
    morton_dense<double, doppler_32_col_mask>  mcb32(32, 32);
    fill_hessian_matrix(mca, 1.0); fill_hessian_matrix(mcb, 2.0); fill_hessian_matrix(mcb32, 2.0);

    // Hybrid row-major
    morton_dense<double,  0x5555555c>      mra(5, 7), mrb(7, 6), mrc(5, 6);
    morton_dense<double, doppler_32_row_mask>  mra32(32, 32), mrc32(32, 32);
    fill_hessian_matrix(mra, 1.0); fill_hessian_matrix(mrb, 2.0); fill_hessian_matrix(mra32, 1.0); 

    mtl::dense2D<double> da(5, 7), db(7, 6), dc(5, 6);
    fill_hessian_matrix(da, 1.0); fill_hessian_matrix(db, 2.0);

    test_pointer(mra32, mcb32, mrc32, "Hybrid col-major and row-major");
 
    test(mda, mdb, mdc, "pure Morton");
    test(da, db, dc, "dense2D");
    test(mra, mrb, mrc, "Hybrid row-major");
    test(mca, mcb, mcc, "Hybrid col-major");
    test(mra, mcb, mrc, "Hybrid col-major and row-major");
    test(mra, db, mrc, "dense2D and row-major");
#endif

    // For measuring:
    // ==============

    unsigned size= 65; 
    if (argc > 1) size= atoi(argv[1]);

    std::cout << "Matrix size " << size << "x" << size << ":\n";

    {
    morton_dense<double,  morton_mask>      mdal(size, size), mdbl(size, size), mdcl(size, size);
    fill_hessian_matrix(mdal, 1.0); fill_hessian_matrix(mdbl, 2.0);
    measure_mult(mdal, mdbl, mdcl, "pure Morton");
    }

    {
    mtl::dense2D<double> dal(size, size), dbl(size, size), dcl(size, size);
    fill_hessian_matrix(dal, 1.0); fill_hessian_matrix(dbl, 2.0);
    measure_mult(dal, dbl, dcl, "dense2D");
    }

    {
    // Hybrid col-major
    morton_dense<double, doppler_32_col_mask>      mcal(size, size), mcbl(size, size), mccl(size, size);
    fill_hessian_matrix(mcal, 1.0); fill_hessian_matrix(mcbl, 2.0);

    // Hybrid row-major
    morton_dense<double, doppler_32_row_mask>      mral(size, size), mrbl(size, size), mrcl(size, size);
    fill_hessian_matrix(mral, 1.0); fill_hessian_matrix(mrbl, 2.0);

    measure_mult(mral, mrbl, mrcl, "Hybrid row-major");
    measure_mult(mcal, mcbl, mccl, "Hybrid col-major");
    measure_mult(mral, mcbl, mrcl, "Hybrid col-major and row-major");
    measure_mult_pointer(mral, mcbl, mrcl, "Hybrid col-major and row-major");
    }
 
    return 0;
}




