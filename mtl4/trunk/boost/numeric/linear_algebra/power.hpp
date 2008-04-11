// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MATH_POWER_INCLUDE
#define MATH_POWER_INCLUDE

#include <concepts>
#include <boost/numeric/linear_algebra/new_concepts.hpp>


namespace math {

    template <typename Op, typename Element, typename Exponent>
        requires std::IntegralLike<Exponent>
              && std::Semiregular<Element>
              && std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	if (n < 1) throw "In power [magma]: exponent must be greater than 0";

	Element value= a;
	for (; n > 1; --n)
	    value= op(value, a);
	return value;
    }


#if 0
    template <typename Op, typename Element, typename Exponent>
        requires SemiGroup<Op, Element> 
              && std::Integral<Exponent>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	return recursive_multiply_and_square(a, n, op);
    }


    template <typename Op, typename Element, typename Exponent>
        requires Monoid<Op, Element> 
              && std::Integral<Exponent>
    {
	return multiply_and_square(a, n, op);
    }
#endif

#if 0 // PIMonoid not yet defined
    template <typename Op, typename Element, typename Exponent>
        requires PIMonoid<Op, Element> 
              && std::Integral<Exponent>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	if (n < 0 && !is_invertible(op, a)) 
	    throw "In power [PIMonoid]: a must be invertible with negative n";

	return n >= 0 ? multiply_and_square(a, n, op) 
	              : multiply_and_square(inverse(op, a), -n, op);
    }
#endif

#if 0
    template <typename Op, typename Element, typename Exponent>
        requires Group<Op, Element> 
              && std::Integral<Exponent>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	return n >= 0 ? multiply_and_square(a, n, op) 
	              : multiply_and_square(inverse(op, a), -n, op);
    }
#endif



} // namespace math

#endif // MATH_POWER_INCLUDE
