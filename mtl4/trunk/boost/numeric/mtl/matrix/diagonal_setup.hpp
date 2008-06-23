// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_DIAGONAL_SETUP_INCLUDE
#define MTL_DIAGONAL_SETUP_INCLUDE

#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl { namespace matrix {

/// Setup a matrix to a multiple of the unity matrix
/** Intended for sparse matrices but works also with dense matrices. 
    If the value is 0 the matrix is only zeroed out, whereby
    a sparse matrix will be empty after this operation,
    i.e. the zeros on the diagonal are not explicitly stored.
    Another special treatment with the value 0 is that the matrix
    does not need to be square.

    Conversely, in order to assign a value different from 0, the matrix
    must be square.
 **/
template <typename Matrix, typename Value>
inline void diagonal_setup(Matrix& matrix, const Value& value)
{
    if (num_rows(matrix) == 0 || num_cols(matrix) == 0) 
	return;

    if (value == math::zero(matrix[0][0])) {
	set_to_zero(matrix);
	return;
    }
    
    MTL_THROW_IF(num_rows(matrix) != num_cols(matrix), matrix_not_square());
    set_to_zero(matrix);
    
    inserter<Matrix>      ins(matrix, 1);

    for (unsigned i= 0; i < num_rows(matrix); i++)
	ins(i, i) << value;
}

}} // namespace mtl::matrix

#endif // MTL_DIAGONAL_SETUP_INCLUDE
