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


void print_time_and_mflops(double time, double size)
{
    std::cout << "    takes " << time << "s = " << 2.0 * size * size * size / time / 1e6f << "MFlops\n";
}


template <typename MatrixA, typename MatrixB, typename MatrixC>
void measure_mult(MatrixA const& a, MatrixB const& b, MatrixC& c,
		 const char* name)
{
    std::cout << "\nMatrix type(s): " << name << "\n";

    //recursion::max_dim_test_static<32>    base_case_test;
    recursion::bound_test_static<32>    base_case_test;

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

    //recursion::max_dim_test_static<32>                   base_case_test;
    recursion::bound_test_static<32>                     base_case_test;
    typedef functor::mult_add_row_times_col_major_32_t   fast_mult_type;

    boost::timer start1;
    recursive_matrix_mult<fast_mult_type, fast_mult_type>(a, b, c, base_case_test);
    print_time_and_mflops(start1.elapsed(), a.num_rows());


    std::cout << "\nOnly recursive part without actual multiplication (MFlops nonsense), Matrix type(s): " 
	      << name << "\n";
    typedef functor::mult_add_empty_t  empty_type;
    // typedef functor::mult_add_empty_t<MatrixA, MatrixB, MatrixC>  empty_type;
    boost::timer start3;
    recursive_matrix_mult<empty_type, empty_type>(a, b, c, base_case_test);
    print_time_and_mflops(start3.elapsed(), a.num_rows());
} 


template <typename MatrixA, typename MatrixB, typename MatrixC>
void measure_mult_pointer_16(MatrixA const& a, MatrixB const& b, MatrixC& c,
				  const char* name)
{
    std::cout << "\nMult with low abstraction, base case 16x16, Matrix type(s): " << name << "\n";

    //recursion::max_dim_test_static<16>                   base_case_test16;
    recursion::bound_test_static<16>                     base_case_test16;
    typedef functor::mult_add_row_times_col_major_16_t   fast_mult_type16;

    boost::timer start2;
    recursive_matrix_mult<fast_mult_type16, fast_mult_type16>(a, b, c, base_case_test16);
    print_time_and_mflops(start2.elapsed(), a.num_rows());
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
 
    {
	// Hybrid col-major
	morton_dense<double, doppler_16_col_mask> mcbl(size, size);
	fill_hessian_matrix(mcbl, 2.0);

	// Hybrid row-major
	morton_dense<double, doppler_16_row_mask>      mral(size, size), mrcl(size, size);
	fill_hessian_matrix(mral, 1.0); 
	
	measure_mult_pointer_16(mral, mcbl, mrcl, "Hybrid col-major and row-major");
    }

    return 0;
}
 
