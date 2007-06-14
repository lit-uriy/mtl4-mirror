// $COPYRIGHT$

#ifndef MTL_DIAGONAL_SETUP_INCLUDE
#define MTL_DIAGONAL_SETUP_INCLUDE

#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl { namespace matrix {

/// Setup a matrix according to a Diagonal equation on a 2D-grid using a five-point-stencil
/** Intended for sparse matrices but works also with dense matrices. Changes the size of
    the matrix if necessary. **/
template <typename Matrix, typename Value>
inline void diagonal_setup(Matrix& matrix, const Value& value)
{
    MTL_THROW_IF(num_rows(matrix) != num_cols(matrix), matrix_not_square());
    set_to_zero(matrix);
    
    if (num_rows(matrix) == 0) return;

    if (value == math::zero(matrix[0][0])) return;

    inserter<Matrix>      ins(matrix, 1);

    for (unsigned i= 0; i < num_rows(matrix); i++)
	ins(i, i) << value;
}

}} // namespace mtl::matrix

#endif // MTL_DIAGONAL_SETUP_INCLUDE
