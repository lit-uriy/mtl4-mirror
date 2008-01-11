// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_ELEMENT_MATRIX_INCLUDE
#define MTL_ELEMENT_MATRIX_INCLUDE

namespace mtl { 

    namespace matrix {

	template <typename Matrix, typename Rows, typename Cols>
	struct element_matrix_t
	{
	    explicit element_matrix_t(const Matrix& matrix, const Rows& rows, const Cols& cols)
		: matrix(matrix), rows(rows), cols(cols)
	    {}

	    const Matrix&  matrix;
	    const Rows&    rows;
	    const Cols&    cols;
	};
    }


template <typename Matrix, typename Rows, typename Cols>
matrix::element_matrix_t<Matrix, Rows, Cols>
inline element_matrix(const Matrix& matrix, const Rows& rows, const Cols& cols)
{
    return matrix::element_matrix_t<Matrix, Rows, Cols>(matrix, rows, cols);
}

template <typename Matrix, typename Rows>
matrix::element_matrix_t<Matrix, Rows, Rows>
inline element_matrix(const Matrix& matrix, const Rows& rows)
{
    return matrix::element_matrix_t<Matrix, Rows, Rows>(matrix, rows, rows);
}


} // namespace mtl

#endif // MTL_ELEMENT_MATRIX_INCLUDE
