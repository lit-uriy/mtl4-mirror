// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_SUBMATRIX_INCLUDE
#define MTL_SUBMATRIX_INCLUDE

namespace mtl {

// Functor type as background for free submatrix function
template <typename Matrix>
struct sub_matrix_t
{
    // typedef  *user_defined*   sub_matrix_type;
    // typedef  *user_defined*   const_sub_matrix_type;
    // typedef  *user_defined*   size_type;
    // sub_matrix_type operator()(Matrix&, size_type, size_type, size_type, size_type);
    // const_sub_matrix_type operator()(Matrix const&, size_type, size_type, size_type, size_type);
};
    

template <typename Matrix>
inline typename sub_matrix_t<Matrix>::sub_matrix_type 
sub_matrix(Matrix& matrix, 
	   typename sub_matrix_t<Matrix>::size_type begin_row, 
	   typename sub_matrix_t<Matrix>::size_type end_row, 
	   typename sub_matrix_t<Matrix>::size_type begin_col, 
	   typename sub_matrix_t<Matrix>::size_type end_col)
{
    return sub_matrix_t<Matrix>()(matrix, begin_row, end_row, begin_col, end_col);
}

template <typename Matrix>
inline typename sub_matrix_t<Matrix>::const_sub_matrix_type 
sub_matrix(Matrix const& matrix, 
	   typename sub_matrix_t<Matrix>::size_type begin_row, 
	   typename sub_matrix_t<Matrix>::size_type end_row, 
	   typename sub_matrix_t<Matrix>::size_type begin_col, 
	   typename sub_matrix_t<Matrix>::size_type end_col)
{
    return sub_matrix_t<Matrix>()(matrix, begin_row, end_row, begin_col, end_col);
}

} // namespace mtl

#endif // MTL_SUBMATRIX_INCLUDE
