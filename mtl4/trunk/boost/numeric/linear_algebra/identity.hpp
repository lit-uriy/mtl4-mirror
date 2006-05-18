// $COPYRIGHT$

#ifndef MATH_IDENTITY_INCLUDE
#define MATH_IDENTITY_INCLUDE

#include <boost/numeric/linear_algebra/operators.hpp>

namespace math {

    template <typename Operator, typename Element>
    struct identity 
    {
	Element operator() (const Element&) const;
    };


    // Additive identity of Element type is by default a converted 0
    // However, for vectors one needs to know the dimension
    // (and in parallel eventually also the distribution).
    // Therefore, an element is passed as reference.
    // It is strongly recommended to specialize this tyypetrait
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

} // namespace math

#endif // MATH_IDENTITY_INCLUDE
