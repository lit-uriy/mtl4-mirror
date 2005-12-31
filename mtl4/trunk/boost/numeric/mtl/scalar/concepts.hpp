// $COPYRIGHT$

#ifndef MTL_CONCEPTS_INCLUDE
#define MTL_CONCEPTS_INCLUDE

#include <bits/concepts.h>
#include <glas/identity.hpp>
#include <glas/is_invertible.hpp>
#include <glas/inverse.hpp>

namespace mtl {

#ifdef __GXX_CONCEPTS__

// Refined version (with inheritance) would look like this
#if 0
template <typename Element, typename Operation>
struct concept Magma
  : std::Callable2<Operation, Element, Element>
{
    where std::Assignable<Element>;

    // Short version
    where result_type == Element;
    
    // Long version if multiple refined concepts have result_type
    // where std::Callable2<Operation, Element, Element>::result_type == Element;
};
#endif

template <typename Element, typename Operation>
struct concept Magma
{
    typename result_type = Element;
    Element operator() (Operation, Element, Element);
    Element const& operator= (Element&, Element const&);
  //Element& operator= (Element&, Element const&);
};

// SemiGroup is a refinement which must be nominal
template <typename Element, typename Operation>
concept SemiGroup
  : Magma<Element, Operation>
{};


template <typename Element, typename Operation>
concept CommutativeSemiGroup
  : SemiGroup<Element, Operation>
{};

// Adding identity
template <typename Element, typename Operation>
concept Monoid
: SemiGroup<Element, Operation> 
{
    Element operator() (glas::identity<Element, Operation>);
};

template <typename Element, typename Operation>
concept CommutativeMonoid
  : Monoid<Element, Operation>
{};


template <typename Element, typename Operation>
concept PartiallyInvertibleMonoid
  : Monoid<Element, Operation> 
{
    bool operator() (glas::is_invertible<Element, Operation>, Element);
};

template <typename Element, typename Operation>
concept PartiallyInvertibleCommutativeMonoid
  : PartiallyInvertibleMonoid<Element, Operation>, CommutativeMonoid<Element, Operation>
{};

template <typename Element, typename Operation>
concept Group
  : PartiallyInvertibleMonoid<Element, Operation>
{
    Element operator() (glas::inverse<Element, Operation>, Element);
};

template <typename Element, typename Operation>
concept AbelianGroup
  : Group<Element, Operation>, PartiallyInvertibleCommutativeMonoid<Element, Operation>
{};


#endif // __GXX_CONCEPTS__

} // namespace mtl

#endif // MTL_CONCEPTS_INCLUDE
