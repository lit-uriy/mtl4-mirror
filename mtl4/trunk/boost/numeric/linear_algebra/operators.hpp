// Copyright 2006. Peter Gottschling, Matthias Troyer, Rolf Bonderer
// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MATH_OPERATORS_INCLUDE
#define MATH_OPERATORS_INCLUDE

#include <functional>


namespace math {

template <typename Element>
struct add : public std::binary_function<Element, Element, Element>
{
    Element operator() (const Element& x, const Element& y)
    {
	return x + y;
    }
};


// Heterogeneous addition, i.e. addends and result type may be different
template <typename A1, typename A2, typename R>
struct heterogeneous_add 
  : public std::binary_function<A1, A2, R>
{
    R operator() (const A1& x, const A2& y)
    {
	return x + y;
    }
};


// The results of char and short additions are int, dito unsigned 
template <> struct add<char> : heterogeneous_add<char, char, int> {};
template <> struct add<short> : heterogeneous_add<short, short, int> {};
template <> struct add<unsigned char> : heterogeneous_add<unsigned char, unsigned char, unsigned int> {};
template <> struct add<unsigned short> : heterogeneous_add<unsigned short, unsigned short, unsigned int> {}; 


template <typename Element>
struct mult : public std::binary_function<Element, Element, Element>
{
    Element operator() (const Element& x, const Element& y)
    {
	return x * y;
    }
};


template <typename A1, typename A2, typename R>
struct heterogeneous_mult 
  : public std::binary_function<A1, A2, R>
{
    R operator() (const A1& x, const A2& y)
    {
	return x * y;
    }
};


// The results of char and short multiplications are int, dito unsigned 
template <> struct mult<char> : heterogeneous_mult<char, char, int> {};
template <> struct mult<short> : heterogeneous_mult<short, short, int> {};
template <> struct mult<unsigned char> : heterogeneous_mult<unsigned char, unsigned char, unsigned int> {};
template <> struct mult<unsigned short> : heterogeneous_mult<unsigned short, unsigned short, unsigned int> {}; 


template <typename Element>
struct min : public std::binary_function<Element, Element, Element>
{
    Element operator() (const Element& x, const Element& y)
    {
	return x <= y ? x : y;
    }
};


template <typename Element>
struct max : public std::binary_function<Element, Element, Element>
{
    Element operator() (const Element& x, const Element& y)
    {
	return x >= y ? x : y;
    }
};


} // namespace math

#endif // MATH_OPERATORS_INCLUDE
