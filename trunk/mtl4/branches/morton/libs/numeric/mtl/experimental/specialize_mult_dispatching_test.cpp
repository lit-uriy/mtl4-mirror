// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

// We define for optimization here to check if dispatching works
// #define MTL_USE_OPTERON_OPTIMIZATION
 
#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>
#include <boost/numeric/mtl/recursion/specialized_matrix_mult.hpp>

 
using namespace std;
using namespace mtl;

template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA& a, MatrixB& b, MatrixC& c, const char* name)
{
    std::cout << "\n" << name << "  --- calling specializing mult:\n";
 
    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 2.0);

    specialized_matrix_mult(a, b, c);

    if (a.num_cols() <= 10) {
	print_matrix_row_cursor(a); print_matrix_row_cursor(b); print_matrix_row_cursor(c); }

    check_hessian_matrix_product(c, a.num_cols());
}
 
int test_main(int argc, char* argv[])
{

#if defined MTL_USE_OPTERON_OPTIMIZATION && defined __GNUC__ && !defined __INTEL_COMPILER
    cout << "optimized on gcc\n";
#else
    cout << "not optimized on gcc\n";
#endif
 

    // Bitmasks:

    const unsigned long morton_mask= generate_mask<true, 0, row_major, 0>::value,
	morton_z_mask= generate_mask<false, 0, row_major, 0>::value,
	doppler_32_row_mask_no_shark= generate_mask<true, 5, row_major, 0>::value,
	doppler_32_col_mask_no_shark= generate_mask<true, 5, col_major, 0>::value,
	doppler_32_row_mask= generate_mask<true, 5, row_major, 1>::value,
	doppler_32_col_mask= generate_mask<true, 5, col_major, 1>::value,
	doppler_z_32_row_mask= generate_mask<false, 5, row_major, 1>::value,
	doppler_z_32_col_mask= generate_mask<false, 5, col_major, 1>::value;
 
    unsigned size= 32; 
    if (argc > 1) size= atoi(argv[1]);

    morton_dense<double,  morton_mask> mda(size, size), mdb(size, size), mdc(size, size);
    mtl::dense2D<double>               da(size, size), db(size, size), dc(size, size);
    morton_dense<double, doppler_32_row_mask_no_shark>      mrans(size, size), mrcns(size, size);;
    morton_dense<double, doppler_32_col_mask_no_shark>      mcbns(size, size);
    morton_dense<double, doppler_32_col_mask>      mca(size, size), mcb(size, size), mcc(size, size);
    morton_dense<double, doppler_32_row_mask>      mra(size, size), mrb(size, size), mrc(size, size);
    morton_dense<double, doppler_z_32_col_mask>    mzca(size, size), mzcb(size, size), mzcc(size, size);
    morton_dense<double, doppler_z_32_row_mask>    mzra(size, size), mzrb(size, size), mzrc(size, size);
    morton_dense<float, doppler_32_col_mask>       mcaf(size, size), mcbf(size, size), mccf(size, size);
    morton_dense<float, doppler_32_row_mask>       mraf(size, size), mrbf(size, size), mrcf(size, size);

    std::cout << "Testing base case optimization\n";
    test(da, db, dc, "dense2D");
    test(mda, mdb, mdc, "pure Morton");
    test(mca, mcb, mcc, "Hybrid col-major");
    test(mra, mrb, mrc, "Hybrid row-major");
    test(mrans, mcbns, mrcns, "Hybrid col-major and row-major, no shark tooth");
    test(mraf, mcbf, mrcf, "Hybrid col-major and row-major with float");
    test(mra, mcb, mrc, "Hybrid col-major and row-major");
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z-order");
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z and E-order");

    return 0;
}
 

