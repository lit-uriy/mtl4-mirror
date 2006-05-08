// $COPYRIGHT$

#ifndef LA_CONCEPTS_INCLUDE
#define LA_CONCEPTS_INCLUDE

#ifdef LA_NO_CONCEPTS
#  warning "Concepts are not used"
#endif

#ifdef LA_WITH_CONCEPTS

#include <bits/concepts.h>
// #include <concepts>

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>


namespace math {



// Concepts for functions mapping to same type or convertible
auto concept UnaryIsoFunction<typename Operation, typename Element>
    //: std::Callable1<Operation, Element>
{
    where std::Callable1<Operation, Element>;
    where std::Convertible<std::Callable1<Operation, Element>::result_type, Element>;

    typename result_type = std::Callable1<Operation, Element>::result_type;
    //    result_type operator()(Operation&, Element);
};

auto concept BinaryIsoFunction<typename Operation, typename Element>
    //: std::Callable2<Operation, Element, Element>
{
    where std::Callable2<Operation, Element, Element>;
    where std::Convertible<std::Callable2<Operation, Element, Element>::result_type, Element>;

    typename result_type = std::Callable2<Operation, Element, Element>::result_type;

    // typename result_type;
    // result_type operator()(Operation&, Element, Element);
};


auto concept Magma<typename Operation, typename Element>
    // : BinaryIsoFunction<Operation, Element>
{
    where std::Assignable<Element>;
    where BinaryIsoFunction<Operation, Element>;

    typename result_type = BinaryIsoFunction<Operation, Element>::result_type;
};


// SemiGroup is a refinement which must be nominal
concept SemiGroup<typename Operation, typename Element>
  : Magma<Operation, Element>
{
    axiom Associativity(Operation op, Element x, Element y, Element z)
    {
	op(x, op(y, z)) == op(op(x, y), z);
    }
};


concept CommutativeSemiGroup<typename Operation, typename Element>
  : SemiGroup<Operation, Element>
{
    axiom Commutativity(Operation op, Element x, Element y)
    {
	op(x, y) == op(y, x);
    }   
};

// Adding identity
concept Monoid<typename Operation, typename Element>
  : SemiGroup<Operation, Element> 
{
    where UnaryIsoFunction< Element, identity<Operation, Element> >;

    axiom Neutralitility(Operation op, Element x)
    {
	op( x, identity<Operation, Element>()(x) ) == x;
	op( identity<Operation, Element>()(x), x ) == x;
    }
};


concept CommutativeCommutativeMonoid<typename Operation, typename Element>
  : SemiGroup<Operation, Element>, Monoid<Operation, Element>
{};


concept PartiallyInvertibleMonoid<typename Operation, typename Element>
  : Monoid<Operation, Element> 
{
    where std::Predicate< is_invertible<Operation, Element>, Element >;
};


concept PartiallyInvertibleCommutativeMonoid<typename Operation, typename Element>
  : PartiallyInvertibleMonoid<Operation, Element>, 
    CommutativeMonoid<Operation, Element>
{};


concept Group<typename Operation, typename Element>
  : PartiallyInvertibleMonoid<Operation, Element>
{
    where UnaryIsoFunction< inverse<Operation, Element>, Element >;
};


concept AbelianGroup<typename Operation, typename Element>
  : Group<Operation, Element>, 
    PartiallyInvertibleCommutativeMonoid<Operation, Element>
{};

} // namespace math


#endif // LA_WITH_CONCEPTS

#endif // LA_CONCEPTS_INCLUDE
