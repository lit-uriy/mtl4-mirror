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

#ifndef MTL_DIAGONAL_SETUP_INCLUDE
#define MTL_DIAGONAL_SETUP_INCLUDE

#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl { namespace matrix {

/// Setup a matrix to a multiple of the unity matrix
/** Intended for sparse matrices but works also with dense matrices. 
    If the value is 0 the matrix is only zeroed out, whereby
    a sparse matrix will be empty after this operation,
    i.e. the zeros on the diagonal are not explicitly stored.
    The diagonal in its generalized form is the set of entries with equal row and column
    index (since r6843, older revision considered it erroneous to store
    a non-zero scalar to a non-square matrix).
 **/
template <typename Matrix, typename Value>
inline void diagonal_setup(Matrix& A, const Value& value)
{
    using math::zero;
    if (num_rows(A) == 0 || num_cols(A) == 0) 
	return;

    set_to_zero(A);

    typename Collection<Matrix>::value_type  ref, my_zero(zero(ref));
    if (value == my_zero)
	return;

    diagonal_setup_finish(A, value, typename traits::category<Matrix>::type());
}

template <typename Matrix, typename Value>
inline void diagonal_setup_finish(Matrix& A, const Value& value, tag::universe)
{
    using std::min; 
    inserter<Matrix>      ins(A, 1);
    for (typename Collection<Matrix>::size_type i= 0, n= min(num_rows(A), num_cols(A)); i < n; ++i)
	ins[i][i] << value;
}

template <typename Matrix, typename Value>
inline void diagonal_setup_finish(Matrix& A, const Value& value, tag::distributed)
{
    typedef typename Collection<Matrix>::size_type size_type;
    typename Matrix::row_distribution_type row_dist(row_distribution(A));
    inserter<Matrix>      ins(A, 1);

    for (size_type i= 0, end= num_rows(local(A)); i < end; ++i) {
	size_type grow= row_dist.local_to_global(i);
	if (grow < num_cols(A))
	    ins[grow][grow] << value;
    }
}

}} // namespace mtl::matrix

#endif // MTL_DIAGONAL_SETUP_INCLUDE
