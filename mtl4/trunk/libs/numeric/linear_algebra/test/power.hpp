// $COPYRIGHT$

#ifndef MTL_POWER_INCLUDE
#define MTL_POWER_INCLUDE

#include <libs/numeric/linear_algebra/test/algebraic_functions.hpp>


namespace mtl {

template <typename Op, typename Element, typename Exponent>
  _GLIBCXX_WHERE( math::Magma<Op, Element> 
            && std::Integral<Exponent> )             // Integral might be lifted
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
    where math::SemiGroup<Op, Element> && std::Integral<Exponent>
inline Element power(const Element& base, Exponent exp, Op op)
{
    // std::cout << "[SemiGroup] ";
    return recursive_multiply_and_square(base, exp, op);
}

template <typename Op, typename Element, typename Exponent>
    where math::Monoid<Op, Element> && std::Integral<Exponent>
inline Element power(const Element& base, Exponent exp, Op op)
{
    // std::cout << "[Monoid] ";
    return multiply_and_square(base, exp, op);
}

template <typename Op, typename Element, typename Exponent>
    where math::PartiallyInvertibleMonoid<Op, Element> && std::SignedIntegral<Exponent>
inline Element power(const Element& base, Exponent exp, Op op)
{
    // std::cout << "[PartiallyInvertibleMonoid] ";
    using math::inverse;

    if (exp < 0 && !is_invertible(op, base)) 
        throw "In power [PartiallyInvertibleMonoid]: base must be invertible with negative exponent";

    return exp >= 0 ? multiply_and_square(base, exp, op) 
	            : multiply_and_square(inverse(op, base), -exp, op);
}

template <typename Op, typename Element, typename Exponent>
    where math::Group<Op, Element> && std::SignedIntegral<Exponent>
inline Element power(const Element& base, Exponent exp, Op op)
{
    // std::cout << "[Group] ";
    using math::inverse;

    return exp >= 0 ? multiply_and_square(base, exp, op) 
	            : multiply_and_square(inverse(op, base), -exp, op);
}


# endif  // __GXX_CONCEPTS__

} // namespace mtl

#endif // MTL_POWER_INCLUDE
