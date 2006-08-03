// $COPYRIGHT$

#ifndef MTL_POWER_EXPLICIT_INCLUDE
#define MTL_POWER_EXPLICIT_INCLUDE

#include <boost/numeric/linear_algebra/algebraic_concepts.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>



namespace mtl {

template <typename Op, typename Element, typename Exponent>
  _GLIBCXX_WHERE( std::Integral<Exponent> 
	    && std::Callable2<Op, Element, Element>
	    && std::Assignable<Element, std::Callable2<Op, Element, Element>::result_type>)            
inline Element power(const Element& base, Exponent exp, Op op) 
{
    if (exp < 1) throw "In power: exponent must be greater than 0";
    // std::cout << "[Magma] ";
    
    Element value= base;
    for (; exp > 1; --exp)
	value= op(value, base);
    return value;
}


# ifndef __GXX_CONCEPTS__
#   ifdef LA_SHOW_WARNINGS
#     warning "Automatic dispatching only works with concept compiler"
#     warning "If structure is a Monoid you can call square_and_multiply directly"
#   endif
# else

template <typename Op, typename Element, typename Exponent>
    where algebra::SemiGroup<Op, Element> && std::Integral<Exponent>
          && std::Callable2<Op, Element, Element>
          && std::Assignable<Element, std::Callable2<Op, Element, Element>::result_type>            
inline Element power(const Element& base, Exponent exp, Op op)
{
    // std::cout << "[SemiGroup] ";

    if (exp <= 0) throw "In recursive_multiply_and_square: exponent must greater than 0";

    Exponent half= exp >> 1;

    // If halt is 0 then exp must be 1 and the result is base
    if (half == 0)
	return base;

    // compute power of downward rounded exponent and square the result
    Element value= power(base, half, op);
    value= op(value, value);

    // if odd another multiplication with base is needed
    if (exp & 1) 
	value= op(value, base);
    return value;
}

// {Op, Element} must be a Monoid
template <typename Op, typename Element, typename Exponent>
    where algebra::Monoid<Op, Element> 
          && std::Integral<Exponent>
          && std::Callable2<Op, Element, Element>
          && std::Assignable<Element, std::Callable2<Op, Element, Element>::result_type>
          && std::Assignable<Element, algebra::Monoid<Op, Element>::identity_result_type>
          && std::Assignable<Element, Element>
inline Element multiply_and_square(const Element& base, Exponent exp, Op op) 
{
    // Same as the simpler form except that the first multiplication is made before 
    // the loop and one squaring is saved this way
    if (exp < 0) throw "In multiply_and_square: negative exponent";

    using math::identity;
    Element value= identity(op, base), square= identity(op, base);

    square= base;
    value= identity(op, base);
    if (exp & 1)
        value= base;

    for (exp>>= 1; exp > 0; exp>>= 1) {
	square= op(square, square); 
	if (exp & 1) 
	    value= op(value, square);
    }
    return value;  
} 

template <typename Op, typename Element, typename Exponent>
    where algebra::Monoid<Op, Element> && std::Integral<Exponent>
          && std::Callable2<Op, Element, Element>
          && std::Assignable<Element, std::Callable2<Op, Element, Element>::result_type>            
          && std::Assignable<Element, algebra::Monoid<Op, Element>::identity_result_type>
          && std::Assignable<Element, Element>
inline Element power(const Element& base, Exponent exp, Op op)
{
    // std::cout << "[Monoid] ";
    return multiply_and_square(base, exp, op);
}

template <typename Op, typename Element, typename Exponent>
    where algebra::Group<Op, Element> && std::SignedIntegral<Exponent>
          && std::Callable2<Op, Element, Element>
          && std::Assignable<Element, std::Callable2<Op, Element, Element>::result_type>            
          && std::Assignable<Element, algebra::Monoid<Op, Element>::identity_result_type>
          && std::Assignable<Element, Element>
inline Element power(const Element& base, Exponent exp, Op op)
{
    // std::cout << "[Group] ";
    using math::inverse;

    return exp >= 0 ? multiply_and_square(base, exp, op) 
	            : multiply_and_square(inverse(op, base), -exp, op);
}


# endif   // __GXX_CONCEPTS__

} // namespace mtl

#endif // MTL_POWER_EXPLICIT_INCLUDE
