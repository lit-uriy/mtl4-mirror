// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

// We define for optimization here to check if dispatching works
// #define MTL_USE_OPTERON_OPTIMIZATION

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/set_to_0.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/specialize_mult_type.hpp>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/operations/opteron/mult_add_base_case_32_shark_2.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/recursion/recursive_matrix_mult.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>


namespace mtl {


template <typename MatrixA, typename MatrixB, typename MatrixC>
void specialized_mult_add(MatrixA const& a, MatrixB const& b, MatrixC& c) 
{
    typedef recursion::bound_test_static<32>                      BaseCaseTest;

    using recursion::base_case_matrix;
    typedef typename base_case_matrix<MatrixA, BaseCaseTest>::type base_a_type;
    typedef typename base_case_matrix<MatrixB, BaseCaseTest>::type base_b_type;
    typedef typename base_case_matrix<MatrixC, BaseCaseTest>::type base_c_type;

    typedef typename mtl::specialize_mult_type<
        MatrixA, MatrixB, MatrixC
      , BaseCaseTest
      , functor::mult_add_simple_t<base_a_type, base_b_type, base_c_type>
    >::type                                                       mult_type;

    // std::cout << "Mult type " << typeid(mult_type()).name() << "\n";

    using recursion::matrix_recurator;
    matrix_recurator<MatrixA>    rec_a(a);
    matrix_recurator<MatrixB>    rec_b(b);
    matrix_recurator<MatrixC>    rec_c(c);
    equalize_depth(rec_a, rec_b, rec_c);

    using recursion::recurator_mult_add;
    recurator_mult_add(rec_a, rec_b, rec_c, mult_type(), BaseCaseTest());
}
 
template <typename MatrixA, typename MatrixB, typename MatrixC>
void specialized_matrix_mult(MatrixA const& a, MatrixB const& b, MatrixC& c)
{
    set_to_0(c);
    specialized_mult_add(a, b, c);
}

} // namespace mtl


 
using namespace std;
using namespace mtl;

template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA& a, MatrixB& b, MatrixC& c, const char* name, bool check)
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

#if defined MTL_USE_OPTERON_OPTIMIZATION && defined __INTEL_COMPILER
  cout << "optimized\n";
#else
  cout << "not optimized\n";
#endif

  // return 0;
 

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
    test(da, db, dc, "dense2D", false);
    test(mda, mdb, mdc, "pure Morton", false);
    test(mca, mcb, mcc, "Hybrid col-major", false);
    test(mra, mrb, mrc, "Hybrid row-major", false);
    test(mrans, mcbns, mrcns, "Hybrid col-major and row-major, no shark tooth", false);
    test(mraf, mcbf, mrcf, "Hybrid col-major and row-major with float", false);
    test(mra, mcb, mrc, "Hybrid col-major and row-major", true);
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z-order", true);
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z and E-order", true);

    return 0;
}
 

