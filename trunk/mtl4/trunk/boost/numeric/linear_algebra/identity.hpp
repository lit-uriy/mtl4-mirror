// $COPYRIGHT$

#ifndef MATH_IDENTITY_INCLUDE
#define MATH_IDENTITY_INCLUDE

#include <boost/numeric/linear_algebra/operators.hpp>

namespace math {

template <typename Operation, typename Element>
struct identity {};


// Additive identity of Element type is by default a converted 0
// However, for vectors one needs to know the dimension
// (and in parallel eventually also the distribution).
// Therefore, an element is passed as reference.
// It is strongly recommended to specialize this functor
// for better efficiency.
template <typename Element>
struct identity< add<Element>, Element > 
{ 
    Element operator() (const Element& ref)
    {
	Element tmp(ref);
	tmp= 0;
	return tmp;
    }
};


// Multiplicative identity of Element type is by default a converted 1
// Same comments as above.
template <typename Element>
struct identity< mult<Element>, Element > 
{ 
    Element operator() (const Element& ref)
    {
	Element tmp(ref);
	tmp= 1;
	return tmp;
    }
};


// Function is shorter than typetrait-like functor
template <typename Element, typename Operation>
Element identity_f(const Element& v)
{
    return identity<Operation, Element>() (v);
}


// Short-cut for multiplicative identity
template <typename Element>
Element zero(const Element& v)
{
    return identity<math::add<Element>, Element>() (v);
}


// Short-cut for multiplicative identity
template <typename Element>
Element one(const Element& v)
{
    return identity<math::mult<Element>, Element>() (v);
}


} // namespace math

#endif // MATH_IDENTITY_INCLUDE
