// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_OPERATORS_INCLUDE
#define MTL_MATRIX_OPERATORS_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/matrix/all_mat_expr.hpp>



namespace mtl { namespace matrix {

template <typename M1, typename M2>
inline mat_mat_plus_expr<M1, M2>
operator+ (const mat_expr<M1>& m1, const mat_expr<M2>& m2)
{
    // do not add matrices with inconsistent value types
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<M1>::type, 
			                typename ashape::ashape<M2>::type>::value));
    return mat_mat_plus_expr<M1, M2>(m1.ref, m2.ref);
}


#if 0
// Planned for future optimizations on sums of dense matrix expressions
template <typename M1, typename M2>
inline dmat_dmat_plus_expr<M1, M2>
operator+ (const dmat_expr<M1>& m1, const dmat_expr<M2>& m2)
{
    // do not add matrices with inconsistent value types
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<M1>::type, 
			                typename ashape::ashape<M2>::type>::value));
    return dmat_dmat_plus_expr<M1, M2>(m1.ref, m2.ref);
}
#endif


template <typename M1, typename M2>
inline mat_mat_minus_expr<M1, M2>
operator- (const mat_expr<M1>& m1, const mat_expr<M2>& m2)
{
    // do not add matrices with inconsistent value types
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<M1>::type, 
			                typename ashape::ashape<M2>::type>::value));
    return mat_mat_minus_expr<M1, M2>(m1.ref, m2.ref);
}



}} // namespace mtl::matrix

#endif // MTL_MATRIX_OPERATORS_INCLUDE
