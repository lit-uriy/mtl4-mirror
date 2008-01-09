// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_SCALE_INCLUDE
#define MTL_SCALE_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/std_concept.hpp>
#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/mtl/vector/map_view.hpp>

namespace mtl { namespace tfunctor {

    // AlgebraicCategory is by default tag::scalar
    template <typename Value1, typename Value2, typename AlgebraicCategory>
    struct scale
    {
	typedef typename Multiplicable<Value1, Value2>::result_type result_type;
	
	explicit scale(const Value1& v1) : v1(v1) {}

	result_type operator() (const Value2& v2) const
	{
	    return v1 * v2;
	}
      private:
	Value1 v1; 
    };


    template <typename Value1, typename Matrix>
    struct scale<Value1, Matrix, tag::matrix>
    {
	typedef matrix::scaled_view<Value1, Matrix> result_type;
	
	explicit scale(const Value1& v1) : v1(v1) {}
	
	result_type operator() (const Matrix& matrix) const
	{
	    return result_type(v1, matrix);
	}
      private:
	Value1 v1; 
    };


    template <typename Value1, typename Vector>
    struct scale<Value1, Vector, tag::vector>
    {
	typedef vector::scaled_view<Value1, Vector> result_type;
	
	explicit scale(const Value1& v1) : v1(v1) {}
	
	result_type operator() (const Vector& vector) const
	{
	    return result_type(v1, vector);
	}
      private:
	Value1 v1; 
    };

} // namespace tfunctor

template <typename Value1, typename Value2>
typename tfunctor::scale<Value1, Value2, typename traits::algebraic_category<Value2>::type>::result_type
inline scale(const Value1& value1, const Value2& value2)
{
    return tfunctor::scale<Value1, Value2, typename traits::algebraic_category<Value2>::type>(value1)(value2);
}

} // namespace mtl

#endif // MTL_SCALE_INCLUDE
