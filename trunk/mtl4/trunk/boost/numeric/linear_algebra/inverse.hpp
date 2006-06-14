// $COPYRIGHT$

#ifndef MATH_INVERSE_INCLUDE
#define MATH_INVERSE_INCLUDE

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

namespace math {

template <typename Operation, typename Element>
struct inverse_t {} ;


template <typename Element>
struct inverse_t< add<Element>, Element >
{ 
    Element operator()(const add<Element>&, const Element& v) const
    { 
	return zero(v) - v; 
    } 
};


template <typename Element>
struct inverse_t< mult<Element>, Element >
{ 
    Element operator()(const mult<Element>&, const Element& v) const
    { 
	return one(v) / v ; 
    } 
};


// Function is shorter than typetrait-like functor
template <typename Operation, typename Element>
inline Element inverse(const Operation& op, const Element& v)
{
    return inverse_t<Operation, Element>() (op, v);
}


// Short-cut for multiplicative inverse
template <typename Element>
inline Element reciprocal(const Element& v)
{
    return inverse(math::mult<Element>(), v);
}

} // namespace math

#endif // MATH_INVERSE_INCLUDE
