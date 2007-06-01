// $COPYRIGHT$

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
