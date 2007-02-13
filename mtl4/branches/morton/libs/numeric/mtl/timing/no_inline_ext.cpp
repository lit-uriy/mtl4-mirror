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


using namespace mtl;
using namespace mtl::recursion; 
using namespace std;  


typedef gen_tiling_44_dense_mat_mat_mult_t<modes::add_mult_assign_t>  tiling_44_base_mult_t;

#if 0
    // Bitmasks: 
    const unsigned long 
	doppler_64_row_mask= generate_mask<true, 6, row_major, 0>::value,
	doppler_64_col_mask= generate_mask<true, 6, col_major, 0>::value;

void hybrid_ext_mult_44(const morton_dense<double,  doppler_64_row_mask>& a, 
			const morton_dense<double,  doppler_64_col_mask>& b,
			morton_dense<double,  doppler_64_row_mask>& c)
{
  tiling_44_base_mult_t()(a, b, c);
}
#endif

void dense_ext_mult_44(const dense2D<double>& a,
		       const dense2D<double, matrix_parameters<col_major> >& b,
		       dense2D<double>& c)
{
  tiling_44_base_mult_t()(a, b, c);
}
 
