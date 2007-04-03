// $COPYRIGHT$

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
};
    
template <typename Value1, typename Value2>
struct minus
{
    typedef typename Subtractable<Value1, Value2>::result_type result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
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
};

template <typename Value1, typename Value2>
struct divide
{
    typedef typename Divisible<Value1, Value2>::result_type result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
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
};
    
template <typename Value1, typename Value2>
struct plus_assign
{
    typedef Value1& result_type;

    static inline result_type apply(Value1& v1, const Value2& v2)
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
};

template <typename Value1, typename Value2>
struct times_assign
{
    typedef Value1& result_type;

    static inline result_type apply(Value1& v1, const Value2& v2)
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
};



}} // namespace mtl::sfunctor

#endif // MTL_SFUNCTOR_INCLUDE
