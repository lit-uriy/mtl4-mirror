// $COPYRIGHT$

#ifndef MTL_SFUNCTOR_INCLUDE
#define MTL_SFUNCTOR_INCLUDE

namespace mtl { namespace sfunctor {

template <typename Value1, typename Value2>
struct plus
{
    // temporary solution
    typedef Value1 result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
    {
	return v1 + v2;
    }
};
    
template <typename Value1, typename Value2>
struct minus
{
    // temporary solution
    typedef Value1 result_type;

    static inline result_type apply(const Value1& v1, const Value2& v2)
    {
	return v1 - v2;
    }
};

template <typename Value1, typename Value2>
struct assign
{
    // temporary solution
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



}} // namespace mtl::sfunctor

#endif // MTL_SFUNCTOR_INCLUDE
