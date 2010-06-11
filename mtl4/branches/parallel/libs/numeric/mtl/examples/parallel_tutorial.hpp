// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_TUTORIAL_INCLUDE
#define MTL_TUTORIAL_INCLUDE

// for references
namespace mtl {

//-----------------------------------------------------------


/*! \page parallel_install Parallel installation guide



Proceed to the \ref parallel_tutorial.  

*/
//-----------------------------------------------------------



//-----------------------------------------------------------
/*! 
\page parallel_testing_scons Parallel testing with scons



*/
//-----------------------------------------------------------



//-----------------------------------------------------------
/*! 
\page parallel_testing_cmake Parallel testing with cmake

In progress.

*/
//-----------------------------------------------------------


//-----------------------------------------------------------

/*! \page parallel_tutorial Parallel tutorial


*/


//-----------------------------------------


//-----------------------------------------------------------
/*! \page parallel_overview_ops Parallel overview

Todo: check which work in parallel

-# %Matrix Operations
   - \subpage mat_vec_expr
   - \subpage adjoint
   - \subpage change_dim
   - \subpage conj
   - \subpage crop
   - \subpage diagonal_setup
   - \subpage eigenvalue_symmetric
   - \subpage extract_hessenberg
   - \subpage extract_householder_hessenberg
   - \subpage frobenius_norm
   - \subpage hermitian
   - \subpage hessenberg
   - \subpage hessenberg_factors
   - \subpage hessenberg_q
   - \subpage hessian_setup
   - \subpage householder_hessenberg
   - \subpage infinity_norm
   - \subpage inv
   - \subpage inv_lower
   - \subpage inv_upper
   - \subpage invert_diagonal
   - \subpage laplacian_setup
   - \subpage lower
   - \subpage lu
   - \subpage lu_p
   - \subpage lu_adjoint_apply
   - \subpage lu_adjoint_solve
   - \subpage lu_apply
   - \subpage lu_f
   - \subpage lu_solve
   - \subpage lu_solve_straight
   - \subpage max_abs_pos
   - \subpage max_pos
   - \subpage num_cols
   - \subpage num_rows
   - \subpage one_norm
   - \subpage op_matrix_equal
   - \subpage op_matrix_add_equal
   - \subpage op_matrix_add
   - \subpage op_matrix_min_equal
   - \subpage op_matrix_min
   - \subpage op_matrix_mult_equal
   - \subpage op_matrix_mult
   - \subpage qr_algo
   - \subpage qr_sym_imp
   - \subpage rank_one_update
   - \subpage rank_two_update
   - \subpage RowInMatrix
   - \subpage set_to_zero
   - \subpage strict_lower
   - \subpage strict_upper
   - \subpage sub_matrix
   - \subpage swap_row
   - \subpage trace
   - \subpage trans
   - \subpage tril
   - \subpage triu
   - \subpage upper
   .
-# %Vector Operations
   - \subpage dot_v
   - \subpage dot_real_v
   - \subpage infinity_norm_v
   - \subpage max_v
   - \subpage min_v
   - \subpage one_norm_v
   - \subpage op_vector_add_equal
   - \subpage op_vector_add
   - \subpage op_vector_min_equal
   - \subpage op_vector_min
   - \subpage orth_v
   - \subpage orth_vi
   - \subpage orthogonalize_factors_v
   - \subpage product_v
   - \subpage size_v
   - \subpage sum_v
   - \subpage swap_row_v
   - \subpage trans_v
   - \subpage two_norm_v
   .
-# %Matrix - %Vector Operations
   - \subpage inverse_lower_trisolve
   - \subpage inverse_upper_trisolve
   - \subpage matrix_vector
   - \subpage lower_trisolve
   - \subpage permutation_av
   - \subpage reorder_av
   - \subpage upper_trisolve
   - \subpage unit_lower_trisolve
   - \subpage unit_upper_trisolve
-# %Scalar - %Vector Operations
   - \subpage scalar_vector_mult_equal
   - \subpage scalar_vector_div_equal
-# miscellaneous
   - \subpage iall
   - \subpage imax
   - \subpage irange

\if Navigation \endif
  Return to \ref performance_athlon &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------

} // namespace mtl


#endif // MTL_TUTORIAL_INCLUDE
