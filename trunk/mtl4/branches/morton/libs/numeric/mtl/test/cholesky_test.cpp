// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp> 
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/operations/assign_modes.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/cholesky.hpp>

using namespace mtl;
using namespace std;  


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


template <typename Matrix>
void print_matrix(Matrix& matrix)
{ 
    using std::cout;
    for (int i=0 ; i<matrix.num_rows(); i++ ){
	for(int j=0; j<matrix.num_cols();  j++ ){
	    cout.fill (' '); cout.width (8); cout.precision (5); cout.flags (ios_base::left);
	    cout << showpoint <<  matrix[i][j] <<"  ";
	}
	cout << endl;
    }
}



template <typename Matrix>
void test(Matrix& matrix, const char* name)
{
    std::cout << "Test " << name << "\n-----\n\n";
    fill_matrix_for_cholesky(matrix);

    recursive_cholesky(matrix);
    if (matrix.num_cols() <= 10) { 
	print_matrix(matrix); std::cout << "\n"; 
    }

    fill_matrix_for_cholesky(matrix);

    with_iterator::recursive_cholesky_base_visitor_t  iter_vis;
    recursive_cholesky(matrix, iter_vis);
    if (matrix.num_cols() <= 10) { 
	print_matrix(matrix); std::cout << "\n"; 
    }

    fill_matrix_for_cholesky(matrix);

    recursive_cholesky_visitor_t<recursion::bound_test_static<2>, with_bracket::cholesky_base_t, with_bracket::tri_solve_base_t, 
                                 with_bracket::tri_schur_base_t, with_bracket::schur_update_base_t>   
        iter_vis2; 
    recursive_cholesky(matrix, iter_vis2);
    if (matrix.num_cols() <= 10) { 
	print_matrix(matrix); std::cout << "\n"; 
    }

    fill_matrix_for_cholesky(matrix);

    recursive_cholesky_visitor_t<recursion::bound_test_static<2>, with_iterator::cholesky_base_t, with_iterator::tri_solve_base_t, 
                                 with_iterator::tri_schur_base_t, with_iterator::schur_update_base_t>   
        iter_vis3;

    recursive_cholesky(matrix, iter_vis3);
    if (matrix.num_cols() <= 10) { 
	print_matrix(matrix); std::cout << "\n"; 
    }


    fill_matrix_for_cholesky(matrix);

    typedef detail::mult_schur_update_t<gen_tiling_22_dense_mat_mat_mult_t<modes::minus_mult_assign_t> > schur_update_22_t;
    recursive_cholesky_visitor_t<recursion::bound_test_static<2>, with_iterator::cholesky_base_t, with_iterator::tri_solve_base_t, 
                                 with_iterator::tri_schur_base_t, schur_update_22_t>   
        iter_vis4;

    recursive_cholesky(matrix, iter_vis4);
    if (matrix.num_cols() <= 10) { 
	print_matrix(matrix); std::cout << "\n"; 
    }


#if 0
    typedef detail::mult_schur_update_t<gen_tiling_44_dense_mat_mat_mult_t<minus_mult_assign_t> > schur_update_44_t;

#endif
}



int test_main(int argc, char* argv[])
{
 
    unsigned size= 13; 
    if (argc > 1) size= atoi(argv[1]); 

    dense2D<double>                                dr(size, size);
    dense2D<double, matrix_parameters<col_major> > dc(size, size);
    morton_dense<double,  morton_mask>             md(size, size);
    morton_dense<double,  morton_z_mask>           mzd(size, size);
    morton_dense<double,  doppler_16_row_mask>     d16r(size, size);
    morton_dense<double,  doppler_32_row_mask>     d32r(size, size);
    morton_dense<double,  doppler_64_row_mask>     d64r(size, size);
    morton_dense<double,  doppler_64_col_mask>     d64c(size, size);
    morton_dense<double,  doppler_128_col_mask>    d128r(size, size);

    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(md, "Morton N-order");
    test(mzd, "Morton Z-order");
    test(d16r, "Hybrid 16 row-major");
    test(d32r, "Hybrid 32 row-major");
    test(d64r, "Hybrid 64 row-major");
    test(d64c, "Hybrid 64 column-major");
    test(d128r, "Hybrid 128 row-major");

    return 0;
}









#if 0
	using glas::tags::row_t; using glas::tags::all_it;
        typedef typename traits::range_generator<row_t, Matrix>::type       cur_type;             
        typedef typename traits::range_generator<all_it, cur_type>::type    iter_type;            
	
	cur_type rb= begin<row_t>(matrix);
	iter_type ib= begin<all_it>(rb);
	ib+= 7;
#endif
