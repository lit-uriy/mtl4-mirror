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

#ifndef MTL_MATRIX_INV_INCLUDE
#define MTL_MATRIX_INV_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/identity.hpp>
#include <boost/numeric/mtl/operation/upper_trisolve.hpp>
#include <boost/numeric/mtl/operation/lu.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>

namespace mtl { namespace matrix {

namespace traits {

    /// Return type of inv(Matrix)
    /** Might be specialized later for the sake of efficiency **/
    template <typename Matrix>
    struct inv
    {
	typedef typename Collection<Matrix>::value_type    value_type;
	typedef ::mtl::matrix::dense2D<value_type>         type;
    };
	
} // traits

namespace detail {

    template <typename Value>
    dense_vector<Value> inline last_unit_vector(std::size_t n)
    {
	using math::zero; using math::one;
	dense_vector<Value> v(n, zero(Value()));
	v[n - 1]= one(Value());
	return v;
    }
}


/// Invert upper triangular matrix
template <typename Matrix>
typename traits::inv<Matrix>::type
inline inv_upper(Matrix const& A)
{
    typedef typename Collection<Matrix>::value_type    value_type;
    typedef typename Collection<Matrix>::size_type     size_type;
   
    size_type N= num_rows(A);
    MTL_THROW_IF(num_cols(A) != N, matrix_not_square());

    typename traits::inv<Matrix>::type Inv(N, N);
    Inv= math::zero(value_type());

    for (size_type k= 0; k < N; ++k) {
	irange r(k+1);
	Inv[r][k]= upper_trisolve(A[r][r], detail::last_unit_vector<value_type>(k+1));
    }
    return Inv;
}

/// Invert lower triangular matrix
template <typename Matrix>
typename traits::inv<Matrix>::type
inline inv_lower(Matrix const& A)
{
    Matrix T(trans(A)); // Shouldn't be needed
    return typename traits::inv<Matrix>::type(trans(inv_upper(T)));
}


/// Invert matrix
/** Uses pivoting LU factorization and triangular inversion
    \sa \ref lu, \ref inv_upper, \ref inv_lower **/
template <typename Matrix>
typename traits::inv<Matrix>::type
inline inv(Matrix const& A)
{
    typedef typename Collection<Matrix>::size_type     size_type;
    typedef typename Collection<Matrix>::value_type    value_type;
    typedef typename traits::inv<Matrix>::type         rt;

    MTL_THROW_IF(num_cols(A) != num_cols(A), matrix_not_square());

    rt                             PLU(A);
    mtl::dense_vector<size_type>   Pv(num_rows(A));

    lu(PLU, Pv);
    rt  PU(upper(PLU)), PL(strict_lower(PLU) + identity<value_type>(num_rows(A), num_cols(A)));

    return rt(inv_upper(PU) * inv_lower(PL) * permutation(Pv));
}


}} // namespace mtl::matrix

#endif // MTL_MATRIX_INV_INCLUDE
