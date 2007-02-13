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
	for (; start.elapsed() < 1; reps++)
	    mult(a, b, c);
	double time= start.elapsed() / double(reps);
	print_time_and_mflops(time, a.num_rows());
	if (time > max_time)
	    enabled[i]= 0;
    } else
	std::cout << ", , ";
}



template <typename Functor, typename Result, typename Arg1, typename Arg2, typename Arg3>
struct no_inline3
{
    Result operator()(Arg1& arg1, Arg2& arg2, Arg3& arg3)
    {
	Functor* f= new(Functor);  // should be scoped pointer
	return apply(f, arg1, arg2, arg3);
    }

    Result apply(Functor* f, Arg1& arg1, Arg2& arg2, Arg3& arg3)
    {
	return (*f)(arg1, arg2, arg3);
    }
};

#if 0
// Specialization not needed, at least not with g++
template <typename Functor, typename Arg1, typename Arg2, typename Arg3>
struct no_inline3<Functor, void, Arg1, Arg2, Arg3>
{
   
    void operator()(Arg1& arg1, Arg2& arg2, Arg3& arg3)
    {
	Functor* f= new(Functor);
	apply(f, arg1, arg2, arg3);
	free(f);
    }

    void apply(Functor* f, Arg1& arg1, Arg2& arg2, Arg3& arg3)
    {
	(*f)(arg1, arg2, arg3);
    }
};
#endif

#if 0
void hybrid_ext_mult_44(const morton_dense<double,  doppler_64_row_mask>& a, 
			const morton_dense<double,  doppler_64_col_mask>& b,
			morton_dense<double,  doppler_64_row_mask>& c);

void dense_ext_mult_44(const dense2D<double>& a,
		       const dense2D<double, matrix_parameters<col_major> >& b,
		       dense2D<double>& c);
 
struct ext_mult_44
{
#if 0
    void operator()(const morton_dense<double,  doppler_64_row_mask>& a, 
		    const morton_dense<double,  doppler_64_col_mask>& b,
		    morton_dense<double,  doppler_64_row_mask>& c)
    {
	hybrid_ext_mult_44(a, b, c);
    }
#endif

    void operator()(const dense2D<double>& a,
		    const dense2D<double, matrix_parameters<col_major> >& b,
		    dense2D<double>& c)
    {
	dense_ext_mult_44(a, b, c);
    }
};

#endif

template <typename Matrix, typename MatrixB> 
void measure_unrolling(unsigned size, std::vector<int>& enabled, Matrix& matrix, MatrixB& matrixb)
{
    std::cout << size << ", ";
 
    gen_recursive_dense_mat_mat_mult_t<base_mult_t>           mult;
    gen_recursive_dense_mat_mat_mult_t<tiling_22_base_mult_t> mult_22;
    gen_recursive_dense_mat_mat_mult_t<tiling_44_base_mult_t> mult_44;

    typedef gen_tiling_22_dense_mat_mat_mult_ft<Matrix, MatrixB, Matrix, modes::add_mult_assign_t> 
      tiling_22_t;
    typedef no_inline3<tiling_22_t, void, const Matrix, const MatrixB, Matrix> tiling_22_no_inline_t;

    typedef gen_tiling_44_dense_mat_mat_mult_ft<Matrix, MatrixB, Matrix, modes::add_mult_assign_t> 
      tiling_44_t;
    typedef no_inline3<tiling_44_t, void, const Matrix, const MatrixB, Matrix> tiling_44_no_inline_t;

    // gen_recursive_dense_mat_mat_mult_t<ext_mult_44>           rec_ext_mult_44;

    typedef typename base_case_matrix<Matrix, test64_t>::type    BaseMatrix;
    typedef typename base_case_matrix<MatrixB, test64_t>::type   BaseMatrixB;
    typedef no_inline3<tiling_44_t, void, const BaseMatrix, const BaseMatrixB, BaseMatrix> tiling_44_base_t;
    gen_recursive_dense_mat_mat_mult_t<tiling_44_base_mult_t>    rec_no_inline_mult_44;
    

    single_measure(matrix, matrixb, matrix, mult, size, enabled, 0);
    single_measure(matrix, matrixb, matrix, mult_22, size, enabled, 1);
    single_measure(matrix, matrixb, matrix, mult_44, size, enabled, 2);
    single_measure(matrix, matrixb, matrix, tiling_22_no_inline_t(), size, enabled, 3);
    single_measure(matrix, matrixb, matrix, tiling_44_no_inline_t(), size, enabled, 4);
    // single_measure(matrix, matrixb, matrix, rec_ext_mult_44, size, enabled, 5);
    single_measure(matrix, matrixb, matrix, rec_no_inline_mult_44, size, enabled, 5);

    std::cout << "0\n";  std::cout.flush();
}

void measure_unrolling_hybrid(unsigned size, std::vector<int>& enabled)
{
    morton_dense<double,  doppler_64_row_mask>     d64r(4, 4);
    morton_dense<double,  doppler_64_col_mask>     d64c(4, 4);
    measure_unrolling(size, enabled, d64r, d64c);
}

void measure_unrolling_dense(unsigned size, std::vector<int>& enabled)
{
    dense2D<double> dense(4, 4);
    dense2D<double, matrix_parameters<col_major> >    b(4, 4);
    measure_unrolling(size, enabled, dense, b);
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
    scenarii.push_back(string("Comparing different unrolling for hybrid row-major matrices"));

    using std::cout;
    if (argc < 3) {
	cerr << "usage: recursive_mult_timing  <steps> <max_size>\n"; 
	exit(1);
    }
    unsigned int steps= atoi(argv[1]), max_size= atoi(argv[2]), size= 32; 
    series(steps, max_size, measure_unrolling_hybrid, scenarii[0]);


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

