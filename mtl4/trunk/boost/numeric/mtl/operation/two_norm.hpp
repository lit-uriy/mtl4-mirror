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

#ifndef MTL_TWO_NORM_INCLUDE
#define MTL_TWO_NORM_INCLUDE

#include <iostream>
#include <cmath>

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/vector/reduction.hpp>
#include <boost/numeric/mtl/vector/reduction_functors.hpp>


namespace mtl {

    namespace impl {

	// Ignore unrolling for matrices 
	template <unsigned long Unroll, typename Matrix>
	typename RealMagnitude<typename Collection<Matrix>::value_type>::type
	inline two_norm(const Matrix& matrix, tag::matrix)
	{
	    std::cout << "Volunteers to implement efficient two-norm of matrices still searched\n";
	    return 0.0;
	}
	
	template <unsigned long Unroll, typename Vector>
	typename RealMagnitude<typename Collection<Vector>::value_type>::type
	inline two_norm(const Vector& vector, tag::vector)
	{
	    using std::sqrt;
	    typedef typename RealMagnitude<typename Collection<Vector>::value_type>::type result_type;
	    return sqrt(vector::reduction<Unroll, vector::two_norm_functor, result_type>::apply(vector));
	}
	
    } // namespace impl


template <unsigned long Unroll, typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline two_norm(const Value& value)
{
    return impl::two_norm<Unroll>(value, typename traits::category<Value>::type());
}

/*! Two-norm for vectors: two_norm(x) \f$\rightarrow |x|_2\f$.
    \retval The magnitude type of the respective value type, see Magnitude.
    The norms are defined as \f$|v|_2=\sqrt{\sum_i |v_i|^2}\f$.

    Vector norms are unrolled 8-fold by default. 
    An n-fold unrolling can be generated with two_norm<n>(x).
    The maximum for n is 8 (it might be increased later).
**/
template <typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline two_norm(const Value& value)
{
    return two_norm<8>(value);
}

} // namespace mtl

#endif // MTL_TWO_NORM_INCLUDE
