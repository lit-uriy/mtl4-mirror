// $COPYRIGHT$

#ifndef MTL_REDUCTION_FUNCTORS_INCLUDE
#define MTL_REDUCTION_FUNCTORS_INCLUDE

#include <cmath>
#include <boost/numeric/linear_algebra/identity.hpp>

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
};


}} // namespace mtl::vector

#endif // MTL_REDUCTION_FUNCTORS_INCLUDE
