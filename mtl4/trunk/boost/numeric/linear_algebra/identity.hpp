// $COPYRIGHT$

#ifndef MATH_IDENTITY_INCLUDE
#define MATH_IDENTITY_INCLUDE

namespace math {

    template <typename Element, typename Operator>
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
    struct identity< Element, add<Element> > 
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
    struct identity< Element, mult<Element> > 
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
