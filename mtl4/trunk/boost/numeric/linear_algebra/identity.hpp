// Copyright 2006. Peter Gottschling, Matthias Troyer, Rolf Bonderer
// $COPYRIGHT$

#ifndef MATH_IDENTITY_INCLUDE
#define MATH_IDENTITY_INCLUDE

#include <boost/numeric/linear_algebra/operators.hpp>
#include <limits>

namespace math {

template <typename Operation, typename Element>
struct identity_t {};

// TBD: Do we the case that the return type is different? Using std::unary_function?

// Additive identity of Element type is by default a converted 0
// However, for vectors one needs to know the dimension
// (and in parallel eventually also the distribution).
// Therefore, an element is passed as reference.
// It is strongly recommended to specialize this functor
// for better efficiency.
template <typename Element>
struct identity_t< add<Element>, Element > 
{ 
    Element operator() (const add<Element>&, const Element& ref) const
    {
	Element tmp(ref);
	tmp= 0;
	return tmp;
    }
};


// Multiplicative identity of Element type is by default a converted 1
// Same comments as above.
// In contrast to additive identity, this default more likely to be wrong (e.g. matrices with all 1s)
template <typename Element>
struct identity_t< mult<Element>, Element > 
{ 
    Element operator() (const mult<Element>&, const Element& ref) const
    {
	Element tmp(ref);
	tmp= 1;
	return tmp;
    }
};


// Identity of max is minimal representable value, for standard types defined in numeric_limits
template <typename Element>
struct identity_t< max<Element>, Element > 
{ 
    Element operator() (const max<Element>&, const Element& ref) const
    {
	using std::numeric_limits;
	return numeric_limits<Element>::min();
    }
};


// Identity of min is maximal representable value, for standard types defined in numeric_limits
template <typename Element>
struct identity_t< min<Element>, Element > 
{ 
    Element operator() (const min<Element>&, const Element& ref) const
    {
	using std::numeric_limits;
	return numeric_limits<Element>::max();
    }
};



// Function is shorter than typetrait-like functor
template <typename Operation, typename Element>
inline Element identity(const Operation& op, const Element& v)
{
    return identity_t<Operation, Element>() (op, v);
}


// Short-cut for additive identity
template <typename Element>
inline Element zero(const Element& v)
{
    return identity_t<math::add<Element>, Element>() (math::add<Element>(), v);
}


// Short-cut for multiplicative identity
template <typename Element>
inline Element one(const Element& v)
{
    return identity_t<math::mult<Element>, Element>() (math::mult<Element>(), v);
}


} // namespace math

#endif // MATH_IDENTITY_INCLUDE
