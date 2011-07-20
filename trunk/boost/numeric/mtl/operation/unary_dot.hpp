// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_UNARY_DOT_INCLUDE
#define MTL_UNARY_DOT_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/vector/reduction.hpp>
#include <boost/numeric/mtl/vector/reduction_functors.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>


namespace mtl {

    namespace vector {

	template <unsigned long Unroll, typename Value>
	typename Collection<Value>::value_type
	inline unary_dot(const Value& value)
	{
	    vampir_trace<2041> tracer;
	    typedef typename Collection<Value>::value_type result_type;
	    return reduction<Unroll, two_norm_functor, result_type>::apply(value);
	}
	
	/*! Dot product of a vector with itself, i.e. unary_dot(v) == dot(v, v).

	    Mathematically, it is also identical with the square of the two_norm.
	    However, unary_dot returns the value_type of v while two_norm yields the
	    RealMagnitude type, thus
	    \code
	      two_norm(v) * two_norm(v) == abs(unary_dot(v))
	    \endcode
	    Internally, the computations are performed in RealMagnitude so that
	    unary_dot(v) is more efficient than dot(v, v) for complex vectors.
	    Furthermore, when the dot product is fused with other expressions,
	    the arguments in dot must be different for the correct semantics of
	    certain fusions.
	    
	    Like vector norms, unary_dot is unrolled 8-fold by default. 
	    An n-fold unrolling can be generated with two_norm<n>(x).
	    The maximum for n is 8 (it might be increased later).
	**/
	template <typename Value>
	typename Collection<Value>::value_type
	inline unary_dot(const Value& value)
	{
	    return unary_dot<8>(value);
	}

	template <unsigned long Unroll, typename Vector>
	struct unary_dot_class
	{
	    typedef typename Collection<Vector>::value_type result_type;

	    unary_dot_class(const Vector& v) : v(v) {}
	    operator result_type() const { return unary_dot(v); }
	    const Vector& v;
	};

	template <unsigned long Unroll, typename Vector>
	unary_dot_class<Unroll, Vector> inline lazy_unary_dot(const Vector& v)
	{
	    return unary_dot_class<Unroll, Vector>(v);
	}

	/// Lazy unary dot product
	/** It is automatically evaluated when (implicitly) converted to result_type which doesn't work in template expressions.
	    Can be used for source-to-source transformations. **/
	template <typename Vector>
	unary_dot_class<8, Vector> inline lazy_unary_dot(const Vector& v)
	{
	    return unary_dot_class<8, Vector>(v);
	}


    } // namespace vector

    using vector::unary_dot;
    using vector::lazy_unary_dot;

} // namespace mtl

#endif // MTL_UNARY_DOT_INCLUDE
