// $COPYRIGHT$

#ifndef MTL_TFUNCTOR_INCLUDE
#define MTL_TFUNCTOR_INCLUDE

#include <boost/numeric/mtl/concept/std_concept.hpp>

/// Namespace for functors with fully typed paramaters
namespace mtl { namespace tfunctor {

template <typename Value1, typename Value2>
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


}} // namespace mtl::tfunctor

#endif // MTL_TFUNCTOR_INCLUDE
