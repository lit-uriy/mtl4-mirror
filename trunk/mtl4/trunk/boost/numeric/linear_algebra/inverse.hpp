// $COPYRIGHT$

#ifndef MATH_INVERSE_INCLUDE
#define MATH_INVERSE_INCLUDE

#include <boost/numeric/linear_algebra/operators.hpp>

namespace math {

    template <typename Element, typename Op>
    struct inverse {} ;


    template <typename Element>
    struct inverse< Element, add<Element> > 
    { 
	Element operator()(const Element& v) 
	{ 
	    return identity< Element, add<Element> >()(v) - v; 
	} 
    };


    template <typename Element>
    struct inverse< Element, mult<Element> > 
    { 
	Element operator()(const Element& v) 
	{ 
	    return identity< Element, mult<Element> >()(v) / v ; 
	} 
    };


} // namespace math

#endif // MATH_INVERSE_INCLUDE
