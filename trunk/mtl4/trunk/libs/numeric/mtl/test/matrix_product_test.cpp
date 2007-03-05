// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

// #define MTL_HAS_BLAS
// #define MTL_USE_OPTERON_OPTIMIZATION

#include <boost/numeric/mtl/utility/glas_tag.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/transposed_view.hpp>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/operation/matrix_mult.hpp>
#include <boost/numeric/mtl/operation/hessian_matrix_utility.hpp>
#include <boost/numeric/mtl/operation/assign_mode.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>

using namespace mtl;
using namespace std;  


template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA& a, MatrixB& b, MatrixC& c, const char* name)
{
    using assign::plus_sum; using assign::minus_sum; 
    using recursion::bound_test_static;

    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 2.0);
  goto recursive;
    std::cout << "\n" << name << "  --- calling simple mult:\n"; std::cout.flush();
    typedef gen_dense_mat_mat_mult_t<>  mult_t;
    mult_t                              mult;

    mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols());

    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    typedef gen_dense_mat_mat_mult_t<plus_sum>  add_mult_t;
    add_mult_t add_mult;

    add_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    typedef gen_dense_mat_mat_mult_t<minus_sum>  minus_mult_t;
    minus_mult_t minus_mult;

    minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);

#if 0
    std::cout << "\n" << name << "  --- calling mult with cursors and property maps:\n"; std::cout.flush();
    gen_cursor_dense_mat_mat_mult_t<>  cursor_mult;

    cursor_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols());

    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    gen_cursor_dense_mat_mat_mult_t<plus_sum>  cursor_add_mult;

    cursor_add_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    gen_cursor_dense_mat_mat_mult_t<minus_sum>  cursor_minus_mult; 

    cursor_minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);
#endif 
    std::cout << "\n" << name << "  --- calling mult with tiling:\n"; std::cout.flush();
    typedef gen_tiling_dense_mat_mat_mult_t<2, 2>  tiling_mult_t;
    tiling_mult_t tiling_mult;

    tiling_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols()); 

    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    typedef gen_tiling_dense_mat_mat_mult_t<2, 2, plus_sum>  tiling_add_mult_t;
    tiling_add_mult_t tiling_add_mult;

    tiling_add_mult(a, b, c); 
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    typedef gen_tiling_dense_mat_mat_mult_t<2, 2, minus_sum>  tiling_minus_mult_t;
    tiling_minus_mult_t tiling_minus_mult;

    tiling_minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);

 
    std::cout << "\n" << name << "  --- calling mult with tiling 2x2:\n"; std::cout.flush();
    typedef gen_tiling_22_dense_mat_mat_mult_t<>  tiling_22_mult_t;
    tiling_22_mult_t tiling_22_mult;

    tiling_22_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols()); 

    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    typedef gen_tiling_22_dense_mat_mat_mult_t<plus_sum>  tiling_22_add_mult_t;
    tiling_22_add_mult_t tiling_22_add_mult;

    tiling_22_add_mult(a, b, c); 
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    typedef gen_tiling_22_dense_mat_mat_mult_t<minus_sum>  tiling_22_minus_mult_t;
    tiling_22_minus_mult_t tiling_22_minus_mult;

    tiling_22_minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);


    std::cout << "\n" << name << "  --- calling mult with tiling 4x4:\n"; std::cout.flush();
    typedef gen_tiling_44_dense_mat_mat_mult_t<>  tiling_44_mult_t;
    tiling_44_mult_t tiling_44_mult;

    tiling_44_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols()); 

    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    typedef gen_tiling_44_dense_mat_mat_mult_t<plus_sum>  tiling_44_add_mult_t;
    tiling_44_add_mult_t tiling_44_add_mult;

    tiling_44_add_mult(a, b, c); 
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    typedef gen_tiling_44_dense_mat_mat_mult_t<minus_sum>  tiling_44_minus_mult_t;
    tiling_44_minus_mult_t tiling_44_minus_mult;

    tiling_44_minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);
 recursive:
    std::cout << "\n" << name << "  --- calling mult recursively:\n"; std::cout.flush();
    // The recursive functor is C= A*B but the base case must be C+= A*B !!!!!!
    gen_recursive_dense_mat_mat_mult_t<add_mult_t, bound_test_static<32> >  recursive_mult;

    recursive_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols()); 

    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    gen_recursive_dense_mat_mat_mult_t<add_mult_t, bound_test_static<32>, plus_sum>  recursive_add_mult;

    recursive_add_mult(a, b, c); 
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    gen_recursive_dense_mat_mat_mult_t<minus_mult_t, bound_test_static<32>, minus_sum>  recursive_minus_mult; 
    recursive_minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);

    std::cout << "\n" << name << "  --- calling mult recursively with tiling:\n"; std::cout.flush();
    // The recursive functor is C= A*B but the base case must be C+= A*B !!!!!!
    gen_recursive_dense_mat_mat_mult_t<tiling_add_mult_t, bound_test_static<32> >  recursive_tiling_mult;

    recursive_tiling_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols()); 
 
    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    gen_recursive_dense_mat_mat_mult_t<tiling_add_mult_t, bound_test_static<32>, plus_sum>  recursive_tiling_add_mult;

    recursive_tiling_add_mult(a, b, c); 
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    gen_recursive_dense_mat_mat_mult_t<tiling_minus_mult_t, bound_test_static<32>, minus_sum>  recursive_tiling_minus_mult; 

    recursive_tiling_minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);

    std::cout << "\n" << name << "  --- calling mult recursively platform specific plus tiling:\n"; std::cout.flush();
    typedef gen_platform_dense_mat_mat_mult_t<plus_sum, tiling_add_mult_t> platform_tiling_add_mult_t;
    gen_recursive_dense_mat_mat_mult_t<platform_tiling_add_mult_t, bound_test_static<32> >  recursive_platform_tiling_mult;
    
    recursive_platform_tiling_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);

#ifdef MTL_HAS_BLAS
    std::cout << "\n" << name << "  --- calling blas mult (empty):\n"; std::cout.flush(); 
    gen_blas_dense_mat_mat_mult_t<>  blas_mult;
    blas_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols()); 
#endif    

#ifdef MTL_USE_OPTERON_OPTIMIZATION
    std::cout << "\n" << name << "  --- calling platform specific mult (empty):\n"; std::cout.flush(); 
    gen_platform_dense_mat_mat_mult_t<>  platform_mult;
    platform_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols());

    std::cout << "\n" << name << "  --- check += :\n"; std::cout.flush();
    gen_platform_dense_mat_mat_mult_t<plus_sum>  platform_add_mult;

    platform_add_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 2.0);
    
    std::cout << "\n" << name << "  --- check -= :\n"; std::cout.flush();
    gen_platform_dense_mat_mat_mult_t<minus_sum>  platform_minus_mult;

    platform_minus_mult(a, b, c);
    check_hessian_matrix_product(c, a.num_cols(), 1.0);

#endif

    if (a.num_cols() <= 0) { 
	print_matrix_row_cursor(a); std::cout << "\n"; print_matrix_row_cursor(b); std::cout << "\n"; 
	print_matrix_row_cursor(c); std::cout << "\n"; } 
}
 



template <typename MatrixA, typename MatrixB, typename MatrixC>
void single_test(MatrixA& a, MatrixB& b, MatrixC& c, const char* name)
{
    using assign::plus_sum; using assign::minus_sum; 
    using recursion::bound_test_static;

    std::cout << "\n\n before matrix multiplication:\n";
    std::cout << "A:\n"; print_matrix_row_cursor(a); 
    std::cout << "B:\n"; print_matrix_row_cursor(b); 
    std::cout << "C:\n"; print_matrix_row_cursor(c); std::cout << "\n"; 

    typedef gen_tiling_dense_mat_mat_mult_t<2, 2, plus_sum>  tiling_add_mult_t;
    tiling_add_mult_t tiling_add_mult;
    tiling_add_mult(a, b, c); 
    
    std::cout << "\n\n after matrix multiplication:\n";
    std::cout << "A:\n"; print_matrix_row_cursor(a); 
    std::cout << "B:\n"; print_matrix_row_cursor(b); 
    std::cout << "C:\n"; print_matrix_row_cursor(c); std::cout << "\n"; 
}

#ifdef MTL_HAS_BLAS
extern "C" {
void dgemm_(const char* transa, const char* transb, 
	    const int* m, const int* n, const int* k,
	    const double* alpha,  const double *da,  const int* lda,
	    const double *db, const int* ldb, const double* dbeta,
	    double *dc, const int* ldc);
}

typedef dense2D<double, matrix_parameters<col_major> >        dc_t;

struct dgemm_t
{
    void operator()(const dc_t& a, const dc_t& b, dc_t& c)
    {
	int size= a.num_rows();
	double alpha= 1.0, beta= 0.0;
	dgemm_("N", "N", &size, &size, &size, &alpha, 
	       const_cast<double*>(&a[0][0]), &size, const_cast<double*>(&b[0][0]), 
	       &size, &beta, &c[0][0], &size);

    }
};


void test_blas()
{
    dense2D<double, matrix_parameters<col_major> > a(7, 7), b(7, 7), c(7, 7);
    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 2.0);
    dgemm_t()(a, b, c);

    print_matrix_row_cursor(c);
    check_hessian_matrix_product(c, a.num_cols());
    
}

#endif // MTL_HAS_BLAS

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

    typedef morton_dense<double, doppler_32_row_mask_no_shark>  morton_t;
    morton_dense<double, doppler_32_row_mask_no_shark>      mrans(size, size), mrbns(size, size), mrcns(size, size);;
    morton_dense<double, doppler_32_col_mask_no_shark>      mcans(size, size), mcbns(size, size), mccns(size, size); 
    morton_dense<double, doppler_32_col_mask>      mca(size, size), mcb(size, size), mcc(size, size);
    morton_dense<double, doppler_32_row_mask>      mra(size, size), mrb(size, size), mrc(size, size);
    morton_dense<double, doppler_z_32_col_mask>    mzca(size, size), mzcb(size, size), mzcc(size, size);
    morton_dense<double, doppler_z_32_row_mask>    mzra(size, size), mzrb(size, size), mzrc(size, size);
    morton_dense<float, doppler_32_col_mask>       mcaf(size, size), mcbf(size, size), mccf(size, size);
    morton_dense<float, doppler_32_row_mask>       mraf(size, size), mrbf(size, size), mrcf(size, size);

    transposed_view<dense2D<double> > trans_db(db); 
    transposed_view<morton_t >        trans_mrbns(mrbns); 

#if 0 
    dense2D<double> ta(1, 2), tb(2, 2), tc(1, 2);
    ta[0][0]= 2.; ta[0][1]= 3.; 
    tb[0][0]= 0.; tb[0][1]= 2.; 
    tb[1][0]= 2.; tb[1][1]= 4.; 
    tc[0][0]= 10.; 
    single_test(ta, tb, tc, "Single test");
#endif

    std::cout << "Testing different products\n";

#if 0
    test(da, trans_db, dc, "dense2D and transposed dense2D");
    test(mrans, trans_mrbns, mrcns, "hybrid with transposed matrix");
#endif

#ifdef MTL_HAS_BLAS
    test_blas();
    return 0;
#endif
  goto hybrid;
    goto hybrid;
    test(da, db, dc, "dense2D");
    test(dca, dcb, dcc, "dense2D col-major");
    test(da, dcb, dc, "dense2D mixed");
    test(fa, fcb, fc, "dense2D mixed, float");
    test(da, fcb, fc, "dense2D mixed, dense and float"); 
    test(mda, mdb, mdc, "pure Morton");
    test(mca, mcb, mcc, "Hybrid col-major");
    test(mra, mrb, mrc, "Hybrid row-major");
 hybrid:
    test(mrans, mcbns, mrcns, "Hybrid col-major and row-major, no shark tooth");
    test(mrans, mrbns, mrcns, "Hybrid row-major, no shark tooth");
    test(mraf, mcbf, mrcf, "Hybrid col-major and row-major with float");
    test(mra, mcb, mrc, "Hybrid col-major and row-major");
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z-order");
    test(mra, mzcb, mzrc, "Hybrid col-major and row-major, Z and E-order");
    test(mra, dcb, mzrc, "Hybrid col-major and row-major, Z and E-order mixed with dense2D");
    test(mra, db, mrcns, "Hybric matrix = Shark * dense2D");
    test(mrans, db, mccns, "Hybric matrix (col-major) = hybrid (row) * dense2D");

    return 0;
}
 














