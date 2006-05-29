// $COPYRIGHT$

#ifndef MATH_INVERSE_INCLUDE
#define MATH_INVERSE_INCLUDE

#include <boost/numeric/linear_algebra/operators.hpp>

namespace math {

template <typename Operation, typename Element>
struct inverse {} ;


template <typename Element>
struct inverse< add<Element>, Element >
{ 
    Element operator()(const Element& v) 
    { 
	return identity< add<Element>, Element >()(v) - v; 
    } 
};


template <typename Element>
struct inverse< mult<Element>, Element >
{ 
    Element operator()(const Element& v) 
    { 
	return identity< mult<Element>, Element >()(v) / v ; 
    } 
};


} // namespace math

#endif // MATH_INVERSE_INCLUDE
