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
#include <boost/numeric/mtl/operations/cholesky.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>
#include <boost/numeric/mtl/operations/assign_modes.hpp>
// #include <boost/numeric/mtl/recursion/recursive_matrix_mult.hpp>

using namespace mtl;
using namespace mtl::recursion; 
using namespace std;  






// Maximum time for a single measurement
// is 10 min
const double max_time= 600;

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
    std::cout << time << ", " << 0.333333333333333333 * size * size * size / time / 1e6f << ", ";
}


// Matrices are only placeholder to provide the type
template <typename Visitor, typename Matrix>
void single_measure(Matrix&, Visitor visitor, unsigned size, std::vector<int>& enabled, int i)
{
    Matrix matrix(size, size);
    fill_matrix_for_cholesky(matrix);

    if (enabled[i]) {
	int reps= 0;
	boost::timer start;	
	for (; start.elapsed() < 5; reps++)
	    recursive_cholesky(matrix, visitor);
	double time= start.elapsed() / double(reps);
	print_time_and_mflops(time, size);
	if (time > max_time)
	    enabled[i]= 0;
    } else
	std::cout << ", , ";
}


template <typename Visitor>
void measure(unsigned size, std::vector<int>& enabled)
{
    
    // The matrices in the following functions are only place holders, the real matrices are used in single_measure
    morton_dense<double,  morton_mask>             md(4, 4);
    morton_dense<double,  morton_z_mask>           mzd(4, 4);
    dense2D<double>                                dr(4, 4);
    dense2D<double, matrix_parameters<col_major> > dc(4, 4);
    morton_dense<double,  doppler_32_row_mask>     d32r(4, 4);
    morton_dense<double,  doppler_64_row_mask>     d64r(4, 4);

    std::cout << size << ", ";

    Visitor visitor;
    single_measure(md, visitor, size, enabled, 0);
    single_measure(mzd, visitor, size, enabled, 1);
    single_measure(dr, visitor, size, enabled, 2);
    single_measure(dc, visitor, size, enabled, 3);
    single_measure(d32r, visitor, size, enabled, 4);
    single_measure(d64r, visitor, size, enabled, 5);
    
    std::cout << "0\n"; // to not finish with comma
    std::cout.flush();
}


template <typename Measure>
void series(unsigned steps, unsigned max_size, Measure measure, const string& comment)
{
    std::cout << "# " << comment << '\n';
    std::cout << "# Gnu-Format size, time, MFlops, ...\n"; 
    std::cout << "# N-order, Z-order, row, col, 32 row, 64 row\n"; std::cout.flush();

    std::vector<int> enabled(16, 1);
    for (unsigned i= steps; i <= max_size; i+= steps)
	measure(i, enabled);
}

int main(int argc, char* argv[])
{

    std::vector<std::string> scenarii;
    scenarii.push_back(string("Comparing canonical implementation with different matrix types"));
    scenarii.push_back(string("Comparing iterator implementation with different matrix types"));
    scenarii.push_back(string("Comparing prev. using fast Schur update (2x2 tiling) with different matrix types"));
    scenarii.push_back(string("Comparing prev. using fast Schur update (4x4 tiling) with different matrix types"));

    using std::cout;
    if (argc < 4) {
	cerr << "usage: recursive_mult_timing <scenario> <steps> <max_size>\nScenarii:\n"; 
	for (unsigned i= 0; i < scenarii.size(); i++)
	    cout << i << ": " << scenarii[i] << "\n";
	exit(1);
    }
    unsigned int scenario= atoi(argv[1]), steps= atoi(argv[2]), max_size= atoi(argv[3]), size= 32; 

    typedef with_bracket::recursive_cholesky_base_visitor_t     bracket_t;
    typedef with_iterator::recursive_cholesky_base_visitor_t    iterator_t;

    typedef detail::mult_schur_update_t<gen_tiling_22_dense_mat_mat_mult_t<modes::minus_mult_assign_t> > schur_update_22_t;
    typedef recursive_cholesky_visitor_t<recursion::bound_test_static<2>, with_iterator::cholesky_base_t, with_iterator::tri_solve_base_t, 
	                                 with_iterator::tri_schur_base_t, schur_update_22_t>   
	tiling_22_t;

    typedef detail::mult_schur_update_t<gen_tiling_44_dense_mat_mat_mult_t<modes::minus_mult_assign_t> > schur_update_44_t;
    typedef recursive_cholesky_visitor_t<recursion::bound_test_static<2>, with_iterator::cholesky_base_t, with_iterator::tri_solve_base_t, 
	                                 with_iterator::tri_schur_base_t, schur_update_44_t>   
	tiling_44_t;


    switch (scenario) {
      case 0: 	series(steps, max_size, measure<bracket_t>, scenarii[0]); break;
      case 1: 	series(steps, max_size, measure<iterator_t>, scenarii[1]); break;
      case 2: 	series(steps, max_size, measure<tiling_22_t>, scenarii[2]); break;
      case 3: 	series(steps, max_size, measure<tiling_44_t>, scenarii[3]); break;
#if 0
      case 4: 	series(steps, max_size, measure_unrolling_hybrid, scenarii[4]); break;
      case 5: 	series(steps, max_size, measure_unrolling_dense, scenarii[5]); break;
      case 6: 	series(steps, max_size, measure_orientation, scenarii[6]); break;
      case 7: 	series(steps, max_size, measure_unrolling_32, scenarii[7]); break;
#endif
    }

    return 0; 

}
 


