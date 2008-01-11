// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_INFINITY_NORM_INCLUDE
#define MTL_INFINITY_NORM_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/operation/max_of_sums.hpp>
#include <boost/numeric/mtl/vector/reduction.hpp>
#include <boost/numeric/mtl/vector/reduction_functors.hpp>


namespace mtl {

    namespace impl {

	// Ignore unrolling for matrices 
	template <unsigned long Unroll, typename Matrix>
	typename RealMagnitude<typename Collection<Matrix>::value_type>::type
	inline infinity_norm(const Matrix& matrix, tag::matrix)
	{
	    typename traits::row<Matrix>::type                             row(matrix); 
	    return impl::max_of_sums(matrix, traits::is_row_major<typename OrientedCollection<Matrix>::orientation>(), 
				     row, num_rows(matrix));
	}

	template <unsigned long Unroll, typename Vector>
	typename RealMagnitude<typename Collection<Vector>::value_type>::type
	inline infinity_norm(const Vector& vector, tag::vector)
	{
	    typedef typename RealMagnitude<typename Collection<Vector>::value_type>::type result_type;
	    return vector::reduction<Unroll, vector::infinity_norm_functor, result_type>::apply(vector);
	}
	
    } // namespace impl


template <unsigned long Unroll, typename Value> 
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline infinity_norm(const Value& value)
{
    return impl::infinity_norm<Unroll>(value, typename traits::category<Value>::type());
}	

/*! Infinity-norm for vectors and matrices: infinity_norm(x) \f$\rightarrow |x|_\infty\f$.
    \retval The magnitude type of the respective value type, see Magnitude.

    The norms are defined
    - For vectors: \f$|v|_\infty=\max_i |v_i|\f$; and
    - For matrices: \f$|A|_\infty=\max_i\{\sum_j(|A_{ij}|)\}\f$.

    Vector norms are unrolled 8-fold by default. 
    An n-fold unrolling can be generated with infinity_norm<n>(x).
    The maximum for n is 8 (it might be increased later).
    Matrix norms are not (yet)
    optimized.
**/
template <typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline infinity_norm(const Value& value)
{
    return infinity_norm<8>(value);
}

} // namespace mtl

#endif // MTL_INFINITY_NORM_INCLUDE
