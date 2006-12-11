// $COPYRIGHT$
 
#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/timer.hpp>

// At the moment optimization takes only place on Opteron compiled with gcc
// the following macro
// #define MTL_USE_OPTERON_OPTIMIZATION
 
#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>
#include <boost/numeric/mtl/recursion/specialized_matrix_mult.hpp>

 
using namespace std;
using namespace mtl;

void print_time_and_mflops(double time, double size)
{ 
    // std::cout << "    takes " << time << "s = " << 2.0 * size * size * size / time / 1e6f << "MFlops\n";
    std::cout << size << ", " << time << ", " << 2.0 * size * size * size / time / 1e6f << "\n";
    std::cout.flush();
}

 
// Matrices are only placeholder to provide the type
template <typename MatrixA, typename MatrixB, typename MatrixC>
double time_measure(MatrixA&, MatrixB&, MatrixC&, unsigned size)
{
    MatrixA a(size, size);
    MatrixB b(size, size);
    MatrixC c(size, size);

    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 2.0); 

    // repeat multiplication if it is less than a second (until it is a second)
    int i; boost::timer start1;
    for (i= 0; start1.elapsed() < 1.0; i++)
	specialized_matrix_mult(a, b, c);
    double elapsed= start1.elapsed() / double(i);
    print_time_and_mflops(elapsed, size);
    return elapsed;
}
 
template <typename MatrixA, typename MatrixB, typename MatrixC>
void time_series(MatrixA& a, MatrixB& b, MatrixC& c, const char* name, unsigned steps, unsigned max_size)
{
    // Maximal time per measurement 20 min
    double max_time= 1200.0;

    std::cout << "\n# " << name << "  --- calling specializing mult:\n";
    std::cout << "# Gnu-Format size, time, MFlops\n";
    std::cout.flush();

    for (unsigned i= steps; i <= max_size; i+= steps) {
	double elapsed= time_measure(a, b, c, i);
	if (elapsed > max_time) break;
    }
}


 
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
 
    unsigned steps= 100, max_size= 4000, size= 32; 
    if (argc > 2) {
	steps= atoi(argv[1]); max_size= atoi(argv[2]);
    }

    // Matrices are actually only place holders for the type
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

    std::cout << "Measuring base case optimization\n";
    time_series(da, db, dc, "dense2D", steps, max_size);
    time_series(mda, mdb, mdc, "pure Morton", steps, max_size);
    time_series(mca, mcb, mcc, "Hybrid col-major", steps, max_size);
    time_series(mra, mrb, mrc, "Hybrid row-major", steps, max_size);
    time_series(mrans, mcbns, mrcns, "Hybrid col-major and row-major, no shark tooth", steps, max_size);
    time_series(mraf, mcbf, mrcf, "Hybrid col-major and row-major with float", steps, max_size);
    time_series(mra, mcb, mrc, "Hybrid col-major and row-major", steps, max_size);
    time_series(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z-order", steps, max_size);
    time_series(mzra, mzcb, mzrc, "Hybrid col-major and row-major, Z and E-order", steps, max_size);

    return 0;
}
