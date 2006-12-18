// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
// #include <boost/test/minimal.hpp>
#include <boost/timer.hpp>

#include <boost/numeric/mtl/matrix_parameters.hpp>
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
    // time and MFlops of single measure
    std::cout << time << ", " << 2.0 * size * size * size / time / 1e6f << ", ";
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

    recursion::bound_test_static<32>    base_case_test;
    std::cout << size << ", ";
    boost::timer start1;
    recursive_matrix_mult_simple(a, b, c, base_case_test); 
    print_time_and_mflops(start1.elapsed(), a.num_rows());

    boost::timer start2;
    recursive_matrix_mult_fast_inner(a, b, c, base_case_test);
    print_time_and_mflops(start2.elapsed(), a.num_rows());

    boost::timer start3;
    recursive_matrix_mult_fast_middle<4, 8>(a, b, c, base_case_test);
    print_time_and_mflops(start3.elapsed(), a.num_rows());

    boost::timer start4;
    recursive_matrix_mult_fast_outer<2, 2, 8>(a, b, c, base_case_test);
    print_time_and_mflops(start4.elapsed(), a.num_rows());

    std::cout << "0\n"; // to not finish with comma
    std::cout.flush();
    return start1.elapsed(); // total time (approx)
}
    
 
template <typename MatrixA, typename MatrixB, typename MatrixC>
void time_series(MatrixA& a, MatrixB& b, MatrixC& c, const string& name, unsigned steps, unsigned max_size)
{
    // Maximal time per measurement 20 min
    double max_time= 1200.0;

    std::cout << "# " << name << "  --- with dispatching recursive multiplication:\n";
    std::cout << "# Gnu-Format size, time, MFlops\n";
    std::cout.flush();

    for (unsigned i= steps; i <= max_size; i+= steps) {
	double elapsed= time_measure(a, b, c, i);
	if (elapsed > max_time) break;
    }
}



int main(int argc, char* argv[])
{
    using std::string;
    std::vector<std::string> scenarii;
    scenarii.push_back(string("Morton Z-order"));
    scenarii.push_back(string("Morton E-order"));
    scenarii.push_back(string("Morton Z/E-order (A,C in Z, B in E-order)"));
    scenarii.push_back(string("Dense row-major"));
    scenarii.push_back(string("Dense col-major"));
    scenarii.push_back(string("Dense row/col-major (R, C, R)"));
    scenarii.push_back(string("Hybrid row-major"));
    scenarii.push_back(string("Hybrid row/col-major (R, C, R)"));
    // scenarii.push_back(string("Z/E Hybrid row/col-major (Z-R, Z-C, E-R)"));

    using std::cout;
    if (argc < 4) {
	cerr << "usage: recursive_mult_timing <scenario> <steps> <max_size>\nScenarii:\n"; 
	for (unsigned i= 0; i < scenarii.size(); i++)
	    cout << i << ": " << scenarii[i] << "\n";
	exit(1);
    }
    unsigned int scenario= atoi(argv[1]), steps= atoi(argv[2]), max_size= atoi(argv[3]), size= 32; 

    // Bitmasks: 
    const unsigned long morton_mask= generate_mask<true, 0, row_major, 0>::value,
	morton_z_mask= generate_mask<false, 0, row_major, 0>::value,
	doppler_32_row_mask= generate_mask<true, 5, row_major, 0>::value,
	doppler_32_col_mask= generate_mask<true, 5, col_major, 0>::value,
	doppler_z_32_row_mask= generate_mask<false, 5, row_major, 1>::value,
	doppler_z_32_col_mask= generate_mask<false, 5, col_major, 1>::value;

    // Matrices are actually only place holders for the type
    mtl::dense2D<double>                           da(size, size), db(size, size), dc(size, size);
    mtl::dense2D<double, matrix_parameters<col_major> >  dca(size, size), dcb(size, size), dcc(size, size);

    morton_dense<double,  morton_mask>             mda(size, size), mdb(size, size), mdc(size, size);
    morton_dense<double,  morton_z_mask>           mzda(size, size), mzdb(size, size), mzdc(size, size);

    morton_dense<double, doppler_32_col_mask>      mca(size, size), mcb(size, size), mcc(size, size);
    morton_dense<double, doppler_32_row_mask>      mra(size, size), mrb(size, size), mrc(size, size);

    morton_dense<double, doppler_z_32_col_mask>    mzca(size, size), mzcb(size, size), mzcc(size, size);
    morton_dense<double, doppler_z_32_row_mask>    mzra(size, size), mzrb(size, size), mzrc(size, size);

    cout << "# Measuring block-recursive computations\n";
    switch (scenario) {
      case 0: 	time_series(mzda, mzdb, mzdc, scenarii[0], steps, max_size); break;
      case 1: 	time_series(mda, mdb, mdc, scenarii[1], steps, max_size); break;
      case 2: 	time_series(mzda, mdb, mzdc, scenarii[2], steps, max_size); break;
      case 3: 	time_series(da, db, dc, scenarii[3], steps, max_size); break;
      case 4: 	time_series(dca, dcb, dcc, scenarii[4], steps, max_size); break;
      case 5: 	time_series(da, dcb, dc, scenarii[5], steps, max_size); break;
      case 6: 	time_series(mra, mrb, mrc, scenarii[6], steps, max_size); break;
      case 7: 	time_series(mra, mcb, mrc, scenarii[7], steps, max_size); break;
	  //case 8: 	time_series(mzra, mzcb, mrc, scenarii[8], steps, max_size); break;
    }








#if 0
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
#endif


#if 0
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
#endif
    return 0;
}
 
