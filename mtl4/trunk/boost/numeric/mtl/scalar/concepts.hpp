// $COPYRIGHT$

#ifndef MTL_CONCEPTS_INCLUDE
#define MTL_CONCEPTS_INCLUDE

#include <bits/concepts.h>
// #include <concepts>

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>

#ifdef __GXX_CONCEPTS__
#  define MTL_WITH_CONCEPTS
#else
#  define MTL_NO_CONCEPTS
#  warning "Concepts are not used"
#endif


namespace math {

#ifdef MTL_WITH_CONCEPTS

// Concepts for functions mapping to same type or convertible
auto concept UnaryIsoFunction<typename Element, typename Operation>
  : std::Callable1<Operation, Element>
{
    where std::Convertible<std::Callable1<Operation, Element>::result_type, Element>;

    typename result_type = std::Callable1<Operation, Element>::result_type;
    //    result_type operator()(Operation&, Element);
};

auto concept BinaryIsoFunction<typename Element, typename Operation>
//: std::Callable2<Operation, Element, Element>
{
    where std::Callable2<Operation, Element, Element>;
    where std::Convertible<std::Callable2<Operation, Element, Element>::result_type, Element>;

    typename result_type = std::Callable2<Operation, Element, Element>::result_type;

    // typename result_type;
    // result_type operator()(Operation&, Element, Element);
};


auto concept Magma<typename Element, typename Operation>
// : BinaryIsoFunction<Element, Operation>
{
    where std::Assignable<Element>;
    where BinaryIsoFunction<Element, Operation>;

    typename result_type = BinaryIsoFunction<Element, Operation>::result_type;
};

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


struct concept Magma<typename Element, typename Operation>
{
    typename result_type = Element;
    Element operator() (Operation, Element, Element);
    Element const& operator= (Element&, Element const&);
  //Element& operator= (Element&, Element const&);
};
#endif

// SemiGroup is a refinement which must be nominal
concept SemiGroup<typename Element, typename Operation>
  : Magma<Element, Operation>
{};


concept CommutativeSemiGroup<typename Element, typename Operation>
  : SemiGroup<Element, Operation>
{};

// Adding identity
concept Monoid<typename Element, typename Operation>
: SemiGroup<Element, Operation> 
{
    where UnaryIsoFunction< Element, identity<Element, Operation> >;

    // typename Element, typename Operation>;

    // Element operator() (glas::identity<Element, Operation>);
};


concept CommutativeMonoid<typename Element, typename Operation>
  : Monoid<Element, Operation>
{};


concept PartiallyInvertibleMonoid<typename Element, typename Operation>
  : Monoid<Element, Operation> 
{
    where std::Predicate< is_invertible<Element, Operation>, Element >;

	// bool operator() (glas::is_invertible<Element, Operation>, Element);
};

concept PartiallyInvertibleCommutativeMonoid<typename Element, typename Operation>
  : PartiallyInvertibleMonoid<Element, Operation>, CommutativeMonoid<Element, Operation>
{};


concept Group<typename Element, typename Operation>
  : PartiallyInvertibleMonoid<Element, Operation>
{
    where UnaryIsoFunction< inverse<Element, Operation>, Element >;

    // Element operator() (glas::inverse<Element, Operation>, Element);
};


concept AbelianGroup<typename Element, typename Operation>
  : Group<Element, Operation>, PartiallyInvertibleCommutativeMonoid<Element, Operation>
{};


#endif // __GXX_CONCEPTS__

} // namespace mtl

#endif // MTL_CONCEPTS_INCLUDE
