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
#include <boost/numeric/linear_algebra/identity.hpp>


namespace math {

    template <typename Op, std::Semiregular Element, Integral Exponent>
        requires std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	std::cout << "[Magma] ";
	if (n < 1) throw "In power [magma]: exponent must be greater than 0";

	Element value= a;
	for (; n > 1; --n)
	    value= op(value, a);
	return value;
    }


    template <typename Op, std::Semiregular Element, Integral Exponent>
        requires SemiGroup<Op, Element> 
              && std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	std::cout << "[SemiGroup] ";
        if (n <= 0) throw "In power [SemiGroup]: exponent must greater than 0";

        Exponent half= n >> 1;

        // If half is 0 then n must be 1 and the result is a
        if (half == 0)
	    return a;

        // compute power of downward rounded exponent and square the result
        Element value= power(a, half, op);
        value= op(value, value);

        // if odd another multiplication with a is needed
        if (n & 1) 
	    value= op(value, a);
        return value;
    }


    template <typename Op, std::Semiregular Element, Integral Exponent>
        requires Monoid<Op, Element> 
              && std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
    inline Element multiply_and_square_horner(const Element& a, Exponent n, Op op) 
    {
	if (n == 0)
	    return Element(identity(op, a));

        if (n <= 0) throw "In multiply_and_square_horner: exponent must be greater than 0";

        // Set mask to highest bit
        Exponent mask= 1 << (8 * sizeof(mask) - 1);

        // If this is a negative number right shift can insert 1s instead of 0s -> infinite loop
        // Therefore we take the 2nd-highest bit
        if (mask < 0)
	    mask= 1 << (8 * sizeof(mask) - 2);

        // find highest 1 bit
        while(!bool(mask & n)) mask>>= 1;

        Element value= a;
        for (mask>>= 1; mask; mask>>= 1) {
	    value= op(value, value);
	    if (n & mask) 
		value= op(value, a);
        }
        return value;
    }
        

    template <typename Op, std::Semiregular Element, Integral Exponent>
        requires Monoid<Op, Element> 
              && std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	std::cout << "[Monoid] ";
	return multiply_and_square_horner(a, n, op);
    }

    template <typename Op, std::Semiregular Element, Integral Exponent>
        requires PIMonoid<Op, Element> 
              && std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	std::cout << "[PIMonoid] ";
	if (n < 0 && !is_invertible(op, a)) 
	    throw "In power [PIMonoid]: a must be invertible with negative exponent";

	if (n < 0)
	    return multiply_and_square_horner(Element(inverse(op, a)), Exponent(-n), op);
	else
	    return multiply_and_square_horner(a, n, op);
    }

#if 1
    template <typename Op, std::Semiregular Element, Integral Exponent>
        requires Group<Op, Element> 
              && std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	std::cout << "[Group] ";
	// For groups we don't need any range test

	if (n < 0)
	    return multiply_and_square_horner(Element(inverse(op, a)), Exponent(-n), op);
	else
	    return multiply_and_square_horner(a, n, op);
    }
#endif


#if 0
    template <typename Op, std::Semiregular Element, typename Exponent>
        requires Group<Op, Element> 
              && Integral<Exponent>
              && std::Callable2<Op, Element, Element>
              && std::Convertible<std::Callable2<Op, Element, Element>::result_type, Element>
              && std::Semiregular<math::Inversion<Op, Element>::result_type>
              && std::HasNegate<Exponent>
              && math::Monoid<Op, math::Inversion<Op, Element>::result_type>
              && Integral< std::HasNegate<Exponent>::result_type>
              && std::Callable2<Op, math::Inversion<Op, Element>::result_type, 
				math::Inversion<Op, Element>::result_type>
              && std::Convertible<std::Callable2<Op, math::Inversion<Op, Element>::result_type, 
						 math::Inversion<Op, Element>::result_type>::result_type, 
				  math::Inversion<Op, Element>::result_type>
    inline Element power(const Element& a, Exponent n, Op op)
    {
	std::cout << "[Group] ";
	// For groups we don't need any range test

	if (n < 0)
	    return multiply_and_square_horner(inverse(op, a), -n, op);
	else
	    return multiply_and_square_horner(a, n, op);
    }
#endif

} // namespace math

#endif // MATH_POWER_INCLUDE
