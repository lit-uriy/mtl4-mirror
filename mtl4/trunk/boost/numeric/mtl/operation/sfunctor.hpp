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

#ifndef MTL_SFUNCTOR_INCLUDE
#define MTL_SFUNCTOR_INCLUDE

#include <boost/numeric/mtl/concept/std_concept.hpp>

namespace mtl { namespace sfunctor {

template <typename Value1, typename Value2>
struct plus
{
    typedef typename Addable<Value1, Value2>::result_type result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
    {
	return v1 + v2;
    }

    result_type operator() (const Value1& v1, const Value2& v2) const
    {
	return v1 + v2;
    }
};
    
template <typename Value1, typename Value2>
struct minus
{
    typedef typename Subtractable<Value1, Value2>::result_type result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
    {
	return v1 - v2;
    }

    result_type operator() (const Value1& v1, const Value2& v2) const
    {
	return v1 - v2;
    }
};

template <typename Value1, typename Value2>
struct times
{
    typedef typename Multiplicable<Value1, Value2>::result_type result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
    {
	return v1 * v2;
    }

    result_type operator() (const Value1& v1, const Value2& v2) const
    {
	return v1 * v2;
    }
};

template <typename Value1, typename Value2>
struct divide
{
    typedef typename Divisible<Value1, Value2>::result_type result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
    {
	return v1 / v2;
    }

    result_type operator() (const Value1& v1, const Value2& v2) const
    {
	return v1 / v2;
    }
};

template <typename Value1, typename Value2>
struct assign
{
    typedef Value1& result_type;

    static inline result_type apply(Value1& v1, const Value2& v2)
    {
	return v1= v2;
    }

    result_type operator() (Value1& v1, const Value2& v2) const
    {
	return v1= v2;
    }
};
    
template <typename Value1, typename Value2>
struct plus_assign
{
    typedef Value1& result_type;

    static inline result_type apply(Value1& v1, const Value2& v2)
    {
	return v1+= v2;
    }

    result_type operator() (Value1& v1, const Value2& v2) const
    {
	return v1+= v2;
    }
};
    
template <typename Value1, typename Value2>
struct minus_assign
{
    typedef Value1& result_type;

    static inline result_type apply(Value1& v1, const Value2& v2)
    {
	return v1-= v2;
    }

    result_type operator() (Value1& v1, const Value2& v2) const
    {
	return v1-= v2;
    }
};

template <typename Value1, typename Value2>
struct times_assign
{
    typedef Value1& result_type;

    static inline result_type apply(Value1& v1, const Value2& v2)
    {
	return v1*= v2;
    }

    result_type operator() (Value1& v1, const Value2& v2) const
    {
	return v1*= v2;
    }
};

template <typename Value1, typename Value2>
struct divide_assign
{
    typedef Value1& result_type;

    static inline result_type apply(Value1& v1, const Value2& v2)
    {
	return v1/= v2;
    }

    result_type operator() (Value1& v1, const Value2& v2) const
    {
	return v1/= v2;
    }
};


// Might be helpful for surplus functor arguments
template <typename Value>
struct identity
{
    typedef Value result_type;

    static inline result_type apply(const Value& v)
    {
	return v;
    }

    result_type operator() (const Value& v)
    {
	return v;
    }
};



}} // namespace mtl::sfunctor

#endif // MTL_SFUNCTOR_INCLUDE
