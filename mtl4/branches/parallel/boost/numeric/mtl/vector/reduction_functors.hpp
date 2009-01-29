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

#ifndef MTL_REDUCTION_FUNCTORS_INCLUDE
#define MTL_REDUCTION_FUNCTORS_INCLUDE

#include <cmath>
#include <functional>
#include <boost/numeric/linear_algebra/identity.hpp>

#ifdef MTL_HAS_MPI
#  include <boost/mpi/operations.hpp>
#endif

namespace mtl { namespace vector {

struct one_norm_functor
{
    template <typename Value>
    static inline void init(Value& value)
    {
	using math::zero;
	value= zero(value);
    }

    template <typename Value, typename Element>
    static inline void update(Value& value, const Element& x)
    {    
	using std::abs;
	value+= abs(x);
    }

    template <typename Value>
    static inline void finish(Value& value, const Value& value2)
    {
	value+= value2;
    }

#ifdef MTL_HAS_MPI
    template <typename Value>
    static inline std::plus<Value> par_reduce(Value) 
    {
	return std::plus<Value>();
    }
#endif
};


// sub-optimal if abs is not needed
struct two_norm_functor
{
    template <typename Value>
    static inline void init(Value& value)
    {
	using math::zero;
	value= zero(value);
    }

    template <typename Value, typename Element>
    static inline void update(Value& value, const Element& x)
    {    
	using std::abs;
	value+= abs(x) * abs(x);
    }

    template <typename Value>
    static inline void finish(Value& value, const Value& value2)
    {
	value+= value2;
    }

#ifdef MTL_HAS_MPI
    template <typename Value>
    static inline std::plus<Value> par_reduce(Value) 
    {
	return std::plus<Value>();
    }
#endif
};


struct infinity_norm_functor
{
    template <typename Value>
    static inline void init(Value& value)
    {
	using math::zero;
	value= zero(value);
    }

    template <typename Value, typename Element>
    static inline void update(Value& value, const Element& x)
    {    
	using std::abs; using std::max;
	value= max(value, abs(x));
    }

    template <typename Value>
    static inline void finish(Value& value, const Value& value2)
    {
	using std::abs; using std::max;
	value= max(value, abs(value2));
    }

#ifdef MTL_HAS_MPI
    template <typename Value>
    static inline boost::mpi::maximum<Value> par_reduce(Value) 
    {
	return boost::mpi::maximum<Value>();
    }
#endif
};


struct sum_functor
{
    template <typename Value>
    static inline void init(Value& value)
    {
	using math::zero;
	value= zero(value);
    }

    template <typename Value, typename Element>
    static inline void update(Value& value, const Element& x)
    {    
	value+= x;
    }

    template <typename Value>
    static inline void finish(Value& value, const Value& value2)
    {
	value+= value2;
    }

#ifdef MTL_HAS_MPI
    template <typename Value>
    static inline std::plus<Value> par_reduce(Value) 
    {
	return std::plus<Value>();
    }
#endif
};


struct product_functor
{
    template <typename Value>
    static inline void init(Value& value)
    {
	using math::one;
	value= one(value);
    }

    template <typename Value, typename Element>
    static inline void update(Value& value, const Element& x)
    {    
	value*= x;
    }

    template <typename Value>
    static inline void finish(Value& value, const Value& value2)
    {
	value*= value2;
    }

#ifdef MTL_HAS_MPI
    template <typename Value>
    static inline std::multiplies<Value> par_reduce(Value) 
    {
	return std::multiplies<Value>();
    }
#endif
};


struct max_functor
{
    template <typename Value>
    static inline void init(Value& value)
    {
	using math::identity; 
	value= identity(math::max<Value>(), value);
    }

    template <typename Value, typename Element>
    static inline void update(Value& value, const Element& x)
    {    
	value= math::max<Value>()(value, x);
    }

    template <typename Value>
    static inline void finish(Value& value, const Value& value2)
    {
	value= math::max<Value>()(value, value2);
    }

#ifdef MTL_HAS_MPI
    template <typename Value>
    static inline boost::mpi::maximum<Value> par_reduce(Value) 
    {
	return boost::mpi::maximum<Value>();
    }
#endif
};


struct min_functor
{
    template <typename Value>
    static inline void init(Value& value)
    {
	using math::identity; 
	value= identity(math::min<Value>(), value);
    }

    template <typename Value, typename Element>
    static inline void update(Value& value, const Element& x)
    {    
	value= math::min<Value>()(value, x);
    }

    template <typename Value>
    static inline void finish(Value& value, const Value& value2)
    {
	value= math::min<Value>()(value, value2);
    }

#ifdef MTL_HAS_MPI
    template <typename Value>
    static inline boost::mpi::minimum<Value> par_reduce(Value) 
    {
	return boost::mpi::minimum<Value>();
    }
#endif
};


}} // namespace mtl::vector

#endif // MTL_REDUCTION_FUNCTORS_INCLUDE
