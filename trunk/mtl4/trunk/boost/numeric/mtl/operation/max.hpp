// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MAX_INCLUDE
#define MTL_MAX_INCLUDE

#include <iostream>
#include <cmath>

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/vector/reduction.hpp>
#include <boost/numeric/mtl/vector/reduction_functors.hpp>


namespace mtl {

    namespace impl {

	// Do we really need this for matrices?
	
	template <unsigned long Unroll, typename Vector>
	typename Collection<Vector>::value_type
	inline max(const Vector& vector, tag::vector)
	{
	    typedef typename Collection<Vector>::value_type result_type;
	    return vector::reduction<Unroll, vector::max_functor, result_type>::apply(vector);
	}
	
    } // namespace impl


template <unsigned long Unroll, typename Value>
typename Collection<Value>::value_type
inline max(const Value& value)
{
    return impl::max<Unroll>(value, typename traits::category<Value>::type());
}

template <typename Value>
typename Collection<Value>::value_type
inline max(const Value& value)
{
    return max<8>(value);
}

} // namespace mtl

#endif // MTL_MAX_INCLUDE
