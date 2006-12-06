// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

// We define for optimization here to check if dispatching works
#ifndef MTL_USE_OPTERON_OPTIMIZATION
#  define MTL_USE_OPTERON_OPTIMIZATION
#endif

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/specialize_mult_type.hpp>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/operations/opteron/mult_add_base_case_32_shark_2.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>


struct gen_mult_t {};

#ifndef __INTEL_COMPILER
namespace mtl { struct mult_add_base_case_32_shark_2_opteron {}; }
#endif

bool is_optimized(const mtl::mult_add_base_case_32_shark_2_opteron& )
{
    return true;
}

bool is_optimized(const gen_mult_t& )
{
    return false;
}

template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA const& a, MatrixB const& b, MatrixC const& c, const char* name, bool check)
{
#if 0    
    typedef mtl::specialize_mult_type<MatrixA, MatrixB, MatrixC,
	mtl::recursion::bound_test_static<32>, gen_mult_t>                 dispatcher;
    std::cout << name << " match a: " << dispatcher::match_a << ", bits: " << dispatcher::base_case_bits << "\n";
#endif
    
    typedef typename mtl::specialize_mult_type<MatrixA, MatrixB, MatrixC,
	mtl::recursion::bound_test_static<32>, gen_mult_t>::type           mult_type;
    bool opt= is_optimized(mult_type());

    std::cout << name << " is optimized = " << opt << ", should be = "
	      << check << "\n";
    if (opt != check) throw "wrong dispatching\n";
}

using namespace std;
using namespace mtl;

int test_main(int argc, char* argv[])
{
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

#ifdef __INTEL_COMPILER
    bool true_on_intel= true;
#else
    bool true_on_intel= false;
#endif 

    std::cout << "Testing base case optimization\n";
    test(da, db, dc, "dense2D", false);
    test(mda, mdb, mdc, "pure Morton", false);
    test(mca, mcb, mcc, "Hybrid col-major", false);
    test(mra, mrb, mrc, "Hybrid row-major", false);
    test(mrans, mcbns, mrcns, "Hybrid col-major and row-major, no shark tooth", false);
    test(mraf, mcbf, mrcf, "Hybrid col-major and row-major with float", false);
    test(mra, mcb, mrc, "Hybrid col-major and row-major", true_on_intel);
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z-order", true_on_intel);
    test(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z and E-order", true_on_intel);

    return 0;
}
 

