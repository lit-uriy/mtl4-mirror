// $COPYRIGHT$

#ifndef MATH_OPERATORS_INCLUDE
#define MATH_OPERATORS_INCLUDE

namespace math {

    template <typename Element>
    struct add
    {
	Element operator() (const Element& x, const Element& y)
	{
	    return x + y;
	}
    };

    template <typename Element>
    struct mult
    {
	Element operator() (const Element& x, const Element& y)
	{
	    return x + y;
	}
    };

    
} // namespace math

#endif // MATH_OPERATORS_INCLUDE
