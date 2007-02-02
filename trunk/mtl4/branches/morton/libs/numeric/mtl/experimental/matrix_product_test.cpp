// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

#define MTL_HAS_BLAS
#define MTL_USE_OPTERON_OPTIMIZATION

#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>

using namespace mtl;
using namespace std;  


template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA& a, MatrixB& b, MatrixC& c, const char* name)
{
 
    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 2.0);

    std::cout << "\n" << name << "  --- calling simple mult:\n"; std::cout.flush();
    gen_dense_mat_mat_mult_t<MatrixA, MatrixB, MatrixC>  mult;

    mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols());

#ifdef MTL_HAS_BLAS
    std::cout << "\n" << name << "  --- calling blas mult (empty):\n"; std::cout.flush(); 
    gen_blas_dense_mat_mat_mult_t<MatrixA, MatrixB, MatrixC>  blas_mult;
    blas_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols());
#endif    

#ifdef MTL_USE_OPTERON_OPTIMIZATION
    std::cout << "\n" << name << "  --- calling platform specific mult (empty):\n"; std::cout.flush(); 
    gen_platform_dense_mat_mat_mult_t<MatrixA, MatrixB, MatrixC>  platform_mult;
    platform_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols());
#endif

    if (a.num_cols() <= 0) {
	print_matrix_row_cursor(a); std::cout << "\n"; print_matrix_row_cursor(b); std::cout << "\n"; 
	print_matrix_row_cursor(c); std::cout << "\n"; }
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
 
    unsigned size= 13; 
    if (argc > 1) size= atoi(argv[1]); 

    dense2D<double>               da(size, size), db(size, size), dc(size, size); 
    dense2D<double, matrix_parameters<col_major> >  dca(size, size), dcb(size, size), dcc(size, size);
    dense2D<float>               fa(size, size), fb(size, size), fc(size, size);
    dense2D<float, matrix_parameters<col_major> >  fca(size, size), fcb(size, size), fcc(size, size);
    morton_dense<double,  morton_mask> mda(size, size), mdb(size, size), mdc(size, size);
    morton_dense<double, doppler_32_row_mask_no_shark>      mrans(size, size), mrcns(size, size);;
    morton_dense<double, doppler_32_col_mask_no_shark>      mcbns(size, size); 
    morton_dense<double, doppler_32_col_mask>      mca(size, size), mcb(size, size), mcc(size, size);
    morton_dense<double, doppler_32_row_mask>      mra(size, size), mrb(size, size), mrc(size, size);
    morton_dense<double, doppler_z_32_col_mask>    mzca(size, size), mzcb(size, size), mzcc(size, size);
    morton_dense<double, doppler_z_32_row_mask>    mzra(size, size), mzrb(size, size), mzrc(size, size);
    morton_dense<float, doppler_32_col_mask>       mcaf(size, size), mcbf(size, size), mccf(size, size);
    morton_dense<float, doppler_32_row_mask>       mraf(size, size), mrbf(size, size), mrcf(size, size);

    std::cout << "Testing different products\n";
    test(da, db, dc, "dense2D"); 
    test(dca, dcb, dcc, "dense2D col-major"); 
    test(da, dcb, dc, "dense2D mixed"); 
    test(fa, fcb, fc, "dense2D mixed, float"); 
    test(da, fcb, fc, "dense2D mixed, dense and float"); 
    test(mda, mdb, mdc, "pure Morton");
    test(mca, mcb, mcc, "Hybrid col-major");
    test(mra, mrb, mrc, "Hybrid row-major");
    test(mrans, mcbns, mrcns, "Hybrid col-major and row-major, no shark tooth");
    test(mraf, mcbf, mrcf, "Hybrid col-major and row-major with float");
    test(mra, mcb, mrc, "Hybrid col-major and row-major");
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z-order");
    test(mra, mzcb, mzrc, "Hybrid col-major and row-major, Z and E-order");
    test(mra, dcb, mzrc, "Hybrid col-major and row-major, Z and E-order mixed with dense2D");

    return 0;
}
 



























#if 0
int test_main(int argc, char* argv[])
{
    //morton_dense<double,  0x55555555>      mda(3, 7), mdb(7, 2), mdc(3, 2);
    morton_dense<double,  0x55555555>      mda(5, 7), mdb(7, 6), mdc(5, 6);
    fill_hessian_matrix(mda, 1.0); fill_hessian_matrix(mdb, 2.0);
    std::cout << "mda:\n";    print_matrix_row_cursor(mda);
    std::cout << "\nmdb:\n";  print_matrix_row_cursor(mdb);
 
    matrix_mult_simple(mda, mdb, mdc);
    std::cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_hessian_matrix_product(mdc, 7);

    mtl::dense2D<double> da(5, 7), db(7, 6), dc(5, 6);
    fill_hessian_matrix(da, 1.0); fill_hessian_matrix(db, 2.0);
    std::cout << "\nda:\n";   print_matrix_row_cursor(da);
    std::cout << "\ndb:\n";   print_matrix_row_cursor(db);

    matrix_mult_simple(da, db, dc);
    std::cout << "\ndc:\n";   print_matrix_row_cursor(dc);
    check_hessian_matrix_product(dc, 7);

    std::cout << "\nNow with fast pseudo dot product\n\n";

#if 0
    matrix_mult_fast_dot(mda, mdb, mdc);
    std::cout << "\nmdc:\n";  print_matrix_row_cursor(mdc);
    check_hessian_matrix_product(mdc, 7);
#endif

    matrix_mult_fast_inner(da, db, dc);
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


#endif

