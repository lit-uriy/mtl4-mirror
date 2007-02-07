// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <boost/timer.hpp>

#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>
#include <boost/numeric/mtl/operations/assign_modes.hpp>
// #include <boost/numeric/mtl/recursion/recursive_matrix_mult.hpp>

using namespace mtl;
using namespace mtl::recursion; 
using namespace std;  

// Maximum time for a single measurement
// is 20 min
const double max_time= 900;

    // Bitmasks: 
    const unsigned long morton_mask= generate_mask<true, 0, row_major, 0>::value,
	morton_z_mask= generate_mask<false, 0, row_major, 0>::value,
	doppler_16_row_mask= generate_mask<true, 4, row_major, 0>::value,
	doppler_16_col_mask= generate_mask<true, 4, col_major, 0>::value,
	doppler_32_row_mask= generate_mask<true, 5, row_major, 0>::value,
	doppler_32_col_mask= generate_mask<true, 5, col_major, 0>::value,
	doppler_z_32_row_mask= generate_mask<false, 5, row_major, 0>::value,
	doppler_z_32_col_mask= generate_mask<false, 5, col_major, 0>::value,
	doppler_64_row_mask= generate_mask<true, 6, row_major, 0>::value,
	doppler_64_col_mask= generate_mask<true, 6, col_major, 0>::value,
	doppler_z_64_row_mask= generate_mask<false, 6, row_major, 0>::value,
	doppler_z_64_col_mask= generate_mask<false, 6, col_major, 0>::value,
	doppler_128_row_mask= generate_mask<true, 7, row_major, 0>::value,
	doppler_128_col_mask= generate_mask<true, 7, col_major, 0>::value,
	shark_32_row_mask= generate_mask<true, 5, row_major, 1>::value,
	shark_32_col_mask= generate_mask<true, 5, col_major, 1>::value,
	shark_z_32_row_mask= generate_mask<false, 5, row_major, 1>::value,
	shark_z_32_col_mask= generate_mask<false, 5, col_major, 1>::value,
	shark_64_row_mask= generate_mask<true, 6, row_major, 1>::value,
	shark_64_col_mask= generate_mask<true, 6, col_major, 1>::value,
	shark_z_64_row_mask= generate_mask<false, 6, row_major, 1>::value,
	shark_z_64_col_mask= generate_mask<false, 6, col_major, 1>::value;

typedef modes::add_mult_assign_t                            ama_t;

typedef recursion::bound_test_static<32>                    test32_t;
typedef recursion::bound_test_static<64>                    test64_t;

typedef gen_dense_mat_mat_mult_t<modes::add_mult_assign_t>  base_mult_t;
typedef gen_recursive_dense_mat_mat_mult_t<base_mult_t>     rec_mult_t;

typedef gen_tiling_22_dense_mat_mat_mult_t<modes::add_mult_assign_t>  tiling_22_base_mult_t;
typedef gen_tiling_44_dense_mat_mat_mult_t<modes::add_mult_assign_t>  tiling_44_base_mult_t;



void print_time_and_mflops(double time, double size)
{
    // time and MFlops of single measure
    std::cout << time << ", " << 2.0 * size * size * size / time / 1e6f << ", ";
}


// Matrices are only placeholder to provide the type
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Mult>
void single_measure(MatrixA&, MatrixB&, MatrixC&, Mult mult, unsigned size, std::vector<int>& enabled, int i)
{
    MatrixA a(size, size);
    MatrixB b(size, size);
    MatrixC c(size, size);
    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 1.0);

    if (enabled[i]) {
	int reps= 0;
	boost::timer start;	
	for (; start.elapsed() < 5; reps++)
	    mult(a, b, c);
	double time= start.elapsed() / double(reps);
	print_time_and_mflops(time, a.num_rows());
	if (time > max_time)
	    enabled[i]= 0;
    } else
	std::cout << ", , ";
}

// The matrices in the following functions are only place holders, the real matrices are used in single_measure
void measure_morton_order(unsigned size, std::vector<int>& enabled)
{
    morton_dense<double,  morton_mask>             mda(4, 4), mdb(4, 4), mdc(4, 4);
    morton_dense<double,  morton_z_mask>           mzda(4, 4), mzdb(4, 4), mzdc(4, 4);
    
    std::cout << size << ", ";
    rec_mult_t  mult;
    single_measure(mda, mdb, mdc, mult, size, enabled, 0);
    single_measure(mzda, mzdb, mzdc, mult, size, enabled, 1);
    single_measure(mda, mzdb, mdc, mult, size, enabled, 2);
    
    std::cout << "0\n"; // to not finish with comma
    std::cout.flush();
}


void measure_cast(unsigned size, std::vector<int>& enabled)
{
    morton_dense<double,  morton_mask>             mda(4, 4), mdb(4, 4), mdc(4, 4);
    morton_dense<double,  doppler_32_row_mask>     d32ra(4, 4), d32rb(4, 4), d32rc(4, 4);
    morton_dense<double,  doppler_64_row_mask>     d64ra(4, 4), d64rb(4, 4), d64rc(4, 4);
    morton_dense<double,  doppler_64_col_mask>     d64ca(4, 4), d64cb(4, 4), d64cc(4, 4); 
    
    rec_mult_t  mult;
    std::cout << size << ", ";
    single_measure(mda, mdb, mdc, mult, size, enabled, 0);
    single_measure(d32ra, d32rb, d32rc, mult, size, enabled, 1);
    single_measure(d64ra, d64rb, d64rc, mult, size, enabled, 2);
    single_measure(d64ca, d64cb, d64cc, mult, size, enabled, 3);
 
    std::cout << "0\n";  std::cout.flush();
}


void measure_with_unroll(unsigned size, std::vector<int>& enabled)
{
    morton_dense<double,  doppler_32_row_mask>     d32r(4, 4);
    morton_dense<double,  doppler_32_col_mask>     d32c(4, 4);

    std::cout << size << ", ";

    gen_recursive_dense_mat_mat_mult_t<base_mult_t, test32_t>           mult;
    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t, test32_t> mult_22;
    gen_recursive_dense_mat_mat_mult_t<tiling_44_base_mult_t, test32_t> mult_44;

    single_measure(d32r, d32r, d32r, mult, size, enabled, 0);
    single_measure(d32r, d32r, d32r, mult_22, size, enabled, 1);
    single_measure(d32r, d32r, d32r, mult_44, size, enabled, 2);

    single_measure(d32c, d32c, d32c, mult, size, enabled, 3);
    single_measure(d32c, d32c, d32c, mult_22, size, enabled, 4);
    single_measure(d32c, d32c, d32c, mult_44, size, enabled, 5);
 
    std::cout << "0\n";  std::cout.flush();
}

void measure_base_size(unsigned size, std::vector<int>& enabled)
{
    morton_dense<double,  doppler_16_row_mask>     d16r(4, 4);
    morton_dense<double,  doppler_32_row_mask>     d32r(4, 4);
    morton_dense<double,  doppler_64_row_mask>     d64r(4, 4);
    morton_dense<double,  doppler_128_col_mask>    d128r(4, 4);
    
    std::cout << size << ", ";

    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t, recursion::bound_test_static<16> > mult16;
    single_measure(d16r, d16r, d16r, mult16, size, enabled, 0);

    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t, recursion::bound_test_static<32> > mult32;
    single_measure(d32r, d32r, d32r, mult32, size, enabled, 1);

    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t, recursion::bound_test_static<64> > mult64;
    single_measure(d64r, d64r, d64r, mult64, size, enabled, 2);

    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t, recursion::bound_test_static<128> > mult128;
    single_measure(d128r, d128r, d128r, mult128, size, enabled, 3);
 
    std::cout << "0\n";  std::cout.flush();
}

template <typename Matrix> 
void measure_unrolling(unsigned size, std::vector<int>& enabled, Matrix& matrix)
{
    std::cout << size << ", ";
 
    gen_recursive_dense_mat_mat_mult_t<base_mult_t>           mult;
    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t> mult_22;
    gen_recursive_dense_mat_mat_mult_t<tiling_44_base_mult_t> mult_44;

    typedef gen_tiling_dense_mat_mat_mult_t<2, 2, ama_t>  tiling_m22_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m22_base_mult_t> mult_m22;
    
    typedef gen_tiling_dense_mat_mat_mult_t<2, 4, ama_t>  tiling_m24_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m24_base_mult_t> mult_m24;

    typedef gen_tiling_dense_mat_mat_mult_t<4, 2, ama_t>  tiling_m42_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m42_base_mult_t> mult_m42;

    typedef gen_tiling_dense_mat_mat_mult_t<3, 5, ama_t>  tiling_m35_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m35_base_mult_t> mult_m35;

    typedef gen_tiling_dense_mat_mat_mult_t<4, 4, ama_t>  tiling_m44_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m44_base_mult_t> mult_m44;


    single_measure(matrix, matrix, matrix, mult, size, enabled, 0);
    single_measure(matrix, matrix, matrix, mult_22, size, enabled, 1);
    single_measure(matrix, matrix, matrix, mult_44, size, enabled, 2);

    single_measure(matrix, matrix, matrix, mult_m22, size, enabled, 3);
    single_measure(matrix, matrix, matrix, mult_m24, size, enabled, 4);
    single_measure(matrix, matrix, matrix, mult_m42, size, enabled, 5);
    single_measure(matrix, matrix, matrix, mult_m35, size, enabled, 6);
    single_measure(matrix, matrix, matrix, mult_m44, size, enabled, 7);
 
    std::cout << "0\n";  std::cout.flush();
}

void measure_unrolling_hybrid(unsigned size, std::vector<int>& enabled)
{
    morton_dense<double,  doppler_64_row_mask>     d64r(4, 4);
    measure_unrolling(size, enabled, d64r);
}

void measure_unrolling_dense(unsigned size, std::vector<int>& enabled)
{
    dense2D<double> dense(4, 4);
    measure_unrolling(size, enabled, dense);
}


void measure_orientation(unsigned size, std::vector<int>& enabled)
{
    std::cout << size << ", ";
 
    morton_dense<double,  doppler_64_row_mask>     d64r(4, 4);
    morton_dense<double,  doppler_64_col_mask>     d64c(4, 4);

    typedef gen_tiling_dense_mat_mat_mult_t<4, 2, ama_t>  tiling_m42_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m42_base_mult_t> mult_m42;

    typedef gen_tiling_dense_mat_mat_mult_t<4, 4, ama_t>  tiling_m44_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m44_base_mult_t> mult_m44;
    
    single_measure(d64r, d64r, d64r, mult_m42, size, enabled, 0);
    single_measure(d64c, d64c, d64c, mult_m42, size, enabled, 1);
    single_measure(d64r, d64c, d64r, mult_m42, size, enabled, 2);
    single_measure(d64c, d64r, d64r, mult_m42, size, enabled, 3);
    
    single_measure(d64r, d64r, d64r, mult_m44, size, enabled, 4);
    single_measure(d64c, d64c, d64c, mult_m44, size, enabled, 5);
    single_measure(d64r, d64c, d64r, mult_m44, size, enabled, 6);
    single_measure(d64c, d64r, d64r, mult_m44, size, enabled, 7);
 
    std::cout << "0\n";  std::cout.flush();
}


void measure_unrolling_32(unsigned size, std::vector<int>& enabled)
{
    std::cout << size << ", ";
 
    gen_recursive_dense_mat_mat_mult_t<base_mult_t, test32_t>           mult;
    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t, test32_t> mult_22;
    gen_recursive_dense_mat_mat_mult_t<tiling_44_base_mult_t, test32_t> mult_44;

    typedef gen_tiling_dense_mat_mat_mult_t<2, 2, ama_t>  tiling_m22_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m22_base_mult_t, test32_t> mult_m22;
    
    typedef gen_tiling_dense_mat_mat_mult_t<2, 4, ama_t>  tiling_m24_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m24_base_mult_t, test32_t> mult_m24;

    typedef gen_tiling_dense_mat_mat_mult_t<4, 2, ama_t>  tiling_m42_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m42_base_mult_t, test32_t> mult_m42;

    typedef gen_tiling_dense_mat_mat_mult_t<3, 5, ama_t>  tiling_m35_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m35_base_mult_t, test32_t> mult_m35;

    typedef gen_tiling_dense_mat_mat_mult_t<4, 4, ama_t>  tiling_m44_base_mult_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_m44_base_mult_t, test32_t> mult_m44;

    
    morton_dense<double,  doppler_32_row_mask>     d32r(4, 4);
    morton_dense<double,  doppler_32_col_mask>     d32c(4, 4);


    single_measure(d32r, d32c, d32r, mult, size, enabled, 0);
    single_measure(d32r, d32c, d32r, mult_22, size, enabled, 1);
    single_measure(d32r, d32c, d32r, mult_44, size, enabled, 2);

    single_measure(d32r, d32c, d32r, mult_m22, size, enabled, 3);
    single_measure(d32r, d32c, d32r, mult_m24, size, enabled, 4);
    single_measure(d32r, d32c, d32r, mult_m42, size, enabled, 5);
    single_measure(d32r, d32c, d32r, mult_m35, size, enabled, 6);
    single_measure(d32r, d32c, d32r, mult_m44, size, enabled, 7);
 
    std::cout << "0\n";  std::cout.flush();
}


template <typename Measure>
void series(unsigned steps, unsigned max_size, Measure measure, const string& comment)
{
    std::cout << "# " << comment << '\n';
    std::cout << "# Gnu-Format size, time, MFlops\n"; std::cout.flush();

    std::vector<int> enabled(16, 1);
    for (unsigned i= steps; i <= max_size; i+= steps)
	measure(i, enabled);
}



int main(int argc, char* argv[])
{

    std::vector<std::string> scenarii;
    scenarii.push_back(string("Comparing Z-, N-order and mixed with recursive multiplication"));
    scenarii.push_back(string("Comparing base case cast (64) for Z-order, hybrid 32, hybrid 64 row and col-major"));
    scenarii.push_back(string("Using unrolled mult on hybrid row- and column-major matrices"));
    scenarii.push_back(string("Comparing base case sizes for corresponding hybrid row-major matrices"));
    scenarii.push_back(string("Comparing different unrolling for hybrid row-major matrices"));
    scenarii.push_back(string("Comparing different unrolling for row-major dense matrices"));
    scenarii.push_back(string("Comparing different orientations for hybrid row-major matrices"));
    scenarii.push_back(string("Comparing different unrolling for hybrid 32 row-major times col-major matrices"));

    using std::cout;
    if (argc < 4) {
	cerr << "usage: recursive_mult_timing <scenario> <steps> <max_size>\nScenarii:\n"; 
	for (unsigned i= 0; i < scenarii.size(); i++)
	    cout << i << ": " << scenarii[i] << "\n";
	exit(1);
    }
    unsigned int scenario= atoi(argv[1]), steps= atoi(argv[2]), max_size= atoi(argv[3]), size= 32; 

    switch (scenario) {
      case 0: 	series(steps, max_size, measure_morton_order, scenarii[0]); break;
      case 1: 	series(steps, max_size, measure_cast, scenarii[1]); break;
      case 2: 	series(steps, max_size, measure_with_unroll, scenarii[2]); break;
      case 3: 	series(steps, max_size, measure_base_size, scenarii[3]); break;
      case 4: 	series(steps, max_size, measure_unrolling_hybrid, scenarii[4]); break;
      case 5: 	series(steps, max_size, measure_unrolling_dense, scenarii[5]); break;
      case 6: 	series(steps, max_size, measure_orientation, scenarii[6]); break;
      case 7: 	series(steps, max_size, measure_unrolling_32, scenarii[7]); break;
    }

    return 0; 

}
 














#if 0

    series(10, 30, measure_morton_order, "Comparing Z-, N-order and mixed with recursive multiplication:");
    series(10, 30, measure_cast, "Comparing base case cast (64) for Z-order, hybrid 32, hybrid 64 row and col-major:");
    series(10, 30, measure_with_unroll, "Using unrolled mult on hybrid row- and column-major matrices:");
    series(10, 30, measure_base_size, "Comparing base case sizes for corresponding hybrid row-major matrices:");
    series(10, 30, measure_unrolling_hybrid, "Comparing different unrolling for hybrid row-major matrices:");
    series(10, 30, measure_unrolling_dense, "Comparing different unrolling for row-major dense matrices:");





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
    scenarii.push_back(string("Z/E Hybrid row/col-major (Z-R, Z-C, E-R)"));
    scenarii.push_back(string("Hybrid 64 row-major"));
    scenarii.push_back(string("Hybrid 64 row/col-major (R, C, R)"));
    scenarii.push_back(string("Z/E Hybrid 64 row/col-major (Z-R, Z-C, E-R)"));
    scenarii.push_back(string("Morton Z-order and Dense col-major"));
    scenarii.push_back(string("Hybrid 64 row-major and dense col-major"));

    using std::cout;
    if (argc < 4) {
	cerr << "usage: recursive_mult_timing <scenario> <steps> <max_size>\nScenarii:\n"; 
	for (unsigned i= 0; i < scenarii.size(); i++)
	    cout << i << ": " << scenarii[i] << "\n";
	exit(1);
    }
    unsigned int scenario= atoi(argv[1]), steps= atoi(argv[2]), max_size= atoi(argv[3]), size= 32; 




    // Matrices are actually only place holders for the type
    mtl::dense2D<double>                           da(size, size), db(size, size), dc(size, size);
    mtl::dense2D<double, matrix_parameters<col_major> >  dca(size, size), dcb(size, size), dcc(size, size);

    morton_dense<double,  morton_mask>             mda(size, size), mdb(size, size), mdc(size, size);
    morton_dense<double,  morton_z_mask>           mzda(size, size), mzdb(size, size), mzdc(size, size);

    morton_dense<double, doppler_32_col_mask>      mca(size, size), mcb(size, size), mcc(size, size);
    morton_dense<double, doppler_32_row_mask>      mra(size, size), mrb(size, size), mrc(size, size);

    morton_dense<double, doppler_z_32_col_mask>    mzca(size, size), mzcb(size, size), mzcc(size, size);
    morton_dense<double, doppler_z_32_row_mask>    mzra(size, size), mzrb(size, size), mzrc(size, size);

    cout << "Measuring block-recursive computations\n";
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
	//case 8: 	time_series(mzda, dcb, dcc, scenarii[8], steps, max_size); break;
    }

    return 0;
#endif





#if 0

// Matrices are only placeholder to provide the type
template <typename MatrixA, typename MatrixB, typename MatrixC>
double time_measure(MatrixA&, MatrixB&, MatrixC&, unsigned size, std::vector<int> enabled)
{
    using std::cout;

    const double max_time= 1200.0;

    MatrixA a(size, size);
    MatrixB b(size, size);
    MatrixC c(size, size);

    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 2.0); 

    typedef recursion::bound_test_static<32>    test32;
    typedef recursion::bound_test_static<64>    test64;

    cout << size << ", ";

    if (enabled[0]) {
	boost::timer start;	
	// do some mult
	print_time_and_mflops(start.elapsed(), a.num_rows());
	if (start.elapsed() > max_time)
	    enabled[0]= false;
    } else
	cout << ", , ";


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
    std::vector<int> enabled(10, true);

    std::cout << "# " << name << "  --- with dispatching recursive multiplication:\n";
    std::cout << "# Gnu-Format size, time, MFlops\n";
    std::cout.flush();

    for (unsigned i= steps; i <= max_size; i+= steps) {
	double elapsed= time_measure(a, b, c, i, enabled);
	if (elapsed > max_time) break;
    }
}
#endif

