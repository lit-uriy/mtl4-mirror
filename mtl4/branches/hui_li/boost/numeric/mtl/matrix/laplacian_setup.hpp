// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_LAPLACIAN_SETUP_INCLUDE
#define MTL_LAPLACIAN_SETUP_INCLUDE

#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>

namespace mtl { namespace matrix {

/// Setup a matrix according to a Laplacian equation on a 2D-grid using a five-point-stencil
/** Intended for sparse matrices but works also with dense matrices. Changes the size of
    the matrix if necessary. **/
template <typename Matrix>
inline void laplacian_setup(Matrix& matrix, unsigned dim1, unsigned dim2)
{
    matrix.change_dim(dim1*dim2, dim1*dim2);
    set_to_zero(matrix);
    inserter<Matrix>      ins(matrix);

    for (unsigned i= 0; i < dim1; i++)
	for (unsigned j= 0; j < dim2; j++) {
	    typename Collection<Matrix>::value_type four(4.0), minus_one(-1.0);
	    unsigned row= i * dim2 + j;
	    ins(row, row) << four;
	    if (j < dim2-1) ins(row, row+1) << minus_one;
	    if (i < dim1-1) ins(row, row+dim2) << minus_one;
	    if (j > 0) ins(row, row-1) << minus_one;
	    if (i > 0) ins(row, row-dim2) << minus_one;
	}
}

}} // namespace mtl::matrix

#endif // MTL_LAPLACIAN_SETUP_INCLUDE
