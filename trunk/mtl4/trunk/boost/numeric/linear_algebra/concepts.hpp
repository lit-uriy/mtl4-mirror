// $COPYRIGHT$

#ifndef LA_CONCEPTS_INCLUDE
#define LA_CONCEPTS_INCLUDE

#include <boost/config/concept_macros.hpp>

#ifdef LA_NO_CONCEPTS
#  warning "Concepts are not used"
#endif

#ifdef LA_WITH_CONCEPTS

#include <bits/concepts.h>
// #include <concepts>

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/linear_algebra/operators.hpp>


namespace math {


// Concepts for functions mapping to same type or convertible
auto concept UnaryIsoFunction<typename Operation, typename Element>
{
    where std::Callable1<Operation, Element>;
    where std::Convertible<std::Callable1<Operation, Element>::result_type, Element>;

    typename result_type = std::Callable1<Operation, Element>::result_type;
};


auto concept BinaryIsoFunction<typename Operation, typename Element>
{
    where std::Callable2<Operation, Element, Element>;
    where std::Convertible<std::Callable2<Operation, Element, Element>::result_type, Element>;

    typename result_type = std::Callable2<Operation, Element, Element>::result_type;
};


auto concept Negatable<typename Element>
{
   typename result_type;
   result_type operator-(Element x);
}

auto concept Magma<typename Operation, typename Element>
    : BinaryIsoFunction<Operation, Element>
{
    where std::Assignable<Element>;
    where std::Assignable<Element, BinaryIsoFunction<Operation, Element>::result_type>;
};


// SemiGroup is a refinement which must be nominal
concept SemiGroup<typename Operation, typename Element>
  : Magma<Operation, Element>
{
    axiom Associativity(Operation op, Element x, Element y, Element z)
    {
	// op(x, op(y, z)) == op(op(x, y), z);              NYS
    }
};


concept CommutativeSemiGroup<typename Operation, typename Element>
  : SemiGroup<Operation, Element>
{
    axiom Commutativity(Operation op, Element x, Element y)
    {
	// op(x, y) == op(y, x);   NYS
    }   
};

// Adding identity
concept Monoid<typename Operation, typename Element>
  : SemiGroup<Operation, Element> 
{
    where UnaryIsoFunction< identity<Operation, Element>, Element >;

    axiom Neutrality(Operation op, Element x)
    {
	// op( x, identity<Operation, Element>()(x) ) == x;   NYS
	// op( identity<Operation, Element>()(x), x ) == x;   NYS
    }
};


concept CommutativeMonoid<typename Operation, typename Element>
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

    axiom Inversion(Operation op, Element x)
    {
	// op( x, inverse<Operation, Element>()(x) ) == identity<Operation, Element>()(x);   NYS
	// op( inverse<Operation, Element>()(x), x ) == identity<Operation, Element>()(x);   NYS
    }
};


concept AbelianGroup<typename Operation, typename Element>
  : Group<Operation, Element>, 
    PartiallyInvertibleCommutativeMonoid<Operation, Element>
{};


// ========================
// Additive scalar concepts
// ========================


auto concept AdditiveMagma<typename Element>
  : Magma< math::add<Element>, Element >
{
    // Operator + 
    where std::Addable<Element>;

    // Operator += by default defined with +, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename result_type;  
    result_type operator+=(Element& x, Element y);
#if 0
    {
	return x= x + y;                      defaults NYS
    }
#endif 

    // Consistency with Magma

    where std::SameType< std::Addable<Element>::result_type, 
	                 Magma< math::add<Element>, Element >::result_type>;
    // or so?
    // where std::MutuallyConvertible< std::Addable<Element>::result_type, 
    //                                 Magma< math::add<Element>, Element >::result_type;

    axiom Consistency(math::add<Element> op, Element x, Element y)
    {
	// op(x, y) == x + y;                    NYS
	// I don't know how to express consistency between + and +=
	// Element tmp = x; tmp+= y; tmp == x + y; 
    }  
}


concept AdditiveSemiGroup<typename Element>
  : AdditiveMagma<Element>, 
    SemiGroup< math::add<Element>, Element >
{};


concept AdditiveCommutativeSemiGroup<typename Element>
  : AdditiveSemiGroup<Element>,
    CommutativeSemiGroup< math::add<Element>, Element >
{};


concept AdditiveMonoid<typename Element>
  : AdditiveSemiGroup<Element>,
    Monoid< math::add<Element>, Element >
{};


// We really need only one of the additive concepts, 
// the requirements of the other would be implied.
// To make the refinement hierarchy clearer, we add them both
concept AdditiveCommutativeMonoid<typename Element>
  : AdditiveMonoid<Element>,
    AdditiveCommutativeSemiGroup<Element>,
    CommutativeMonoid< math::add<Element>, Element >
{};




concept AdditiveGroup<typename Element>
  : AdditiveMonoid<Element>,
    Group< math::add<Element>, Element >
{
    // Operator -, binary and unary
    where std::Subtractable<Element>;   
    where Negatable<Element>;

    // Operator -= by default defined with -, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename result_type;  
    result_type operator-=(Element& x, Element y);
#if 0
    {
	return x= x - y;                      defaults NYS
    }
#endif 
     
    result_type operator-(Element x);
#if 0
    {
	return identity<math::add<Element>, Element>()(x) - x;      defaults NYS
    }
#endif 
    
    axiom Consistency(math::add<Element> op, Element x, Element y)
    {
	// consistency between Group and AdditiveGroup
	// op(x, inverse<math::add<Element>, Element>() (y)) == x - y;                    NYS
	// inverse<math::add<Element>, Element>() (y) == -y;

	// consistency between unary and binary -
	// Element(0) - x == -x

	// I don't know how to express consistency between - and -=
    }  
};


concept AdditiveAbelianGroup<typename Element>
  : AdditiveGroup<Element>,
    AdditiveCommutativeMonoid<Element>,
    AbelianGroup< math::add<Element>, Element >
{};


// ============================
// Multiplitive scalar concepts
// ============================


auto concept MultiplicativeMagma<typename Element>
  : Magma< math::mult<Element>, Element >
{
    // Operator + 
    where std::Multiplicable<Element>;

    // Operator += by default defined with +, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename result_type;  
    result_type operator*=(Element& x, Element y);
#if 0
    {
	return x= x * y;                      defaults NYS
    }
#endif 

    // Consistency with Magma

    where std::SameType< std::Addable<Element>::result_type, 
	                 Magma< math::mult<Element>, Element >::result_type>;
    // or so?
    // where std::MutuallyConvertible< std::Addable<Element>::result_type, 
    //                                 Magma< math::mult<Element>, Element >::result_type;

    axiom Consistency(math::mult<Element> op, Element x, Element y)
    {
	// op(x, y) == x * y;                    NYS
	// I don't know how to express consistency between * and *=
	// Element tmp = x; tmp*= y; tmp == x * y; 
    }  
}


concept MultiplicativeSemiGroup<typename Element>
  : MultiplicativeMagma<Element>, 
    SemiGroup< math::mult<Element>, Element >
{};


concept MultiplicativeCommutativeSemiGroup<typename Element>
  : MultiplicativeSemiGroup<Element>,
    CommutativeSemiGroup< math::mult<Element>, Element >
{};


concept MultiplicativeMonoid<typename Element>
  : MultiplicativeSemiGroup<Element>,
    Monoid< math::mult<Element>, Element >
{};


// We really need only one of the multiplicative concepts, 
// the requirements of the other would be implied.
// To make the refinement hierarchy clearer, we add them both
concept MultiplicativeCommutativeMonoid<typename Element>
  : MultiplicativeMonoid<Element>,
    MultiplicativeCommutativeSemiGroup<Element>,
    CommutativeMonoid< math::mult<Element>, Element >
{};


auto concept DividableWithAssign<typename T, typename U = T>
{
    typename result_type;
    result_type operator/(const Element& t, const Element& u);

    // Operator -= by default defined with -, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename result_type;  
    result_type operator/=(Element& x, Element y);
#if 0
    {
	return x= x / y;                      defaults NYS
    }
#endif 
}    


concept MultiplicativeGroup<typename Element>
  : MultiplicativeMonoid<Element>,
    Group< math::mult<Element>, Element >,
    DividableWithAssign<Element>
{        
    axiom Consistency(math::mult<Element> op, Element x, Element y)
    {
	// consistency between Group and MultiplicativeGroup
	// op(x, inverse<math::mult<Element>, Element>() (y)) == x / y;                    NYS

	// I don't know how to express consistency between / and /=
    }  
};


concept MultiplicativeAbelianGroup<typename Element>
  : MultiplicativeGroup<Element>,
    MultiplicativeCommutativeMonoid<Element>,
    AbelianGroup< math::mult<Element>, Element >
{};


// ======================================
// Algebraic concepts with two connectors
// ======================================


// Alternative definitions use MultiplicativeMonoid<Element> for Ring
// and call such concepts Pseudo-Ring

concept Ring<typename Element>
  : AdditiveAbelianGroup<Element>,
    MultiplicativeSemiGroup<Element>
{
    axiom Distributivity(Element x, Element y, Element z)
    {
	// From left
	// x * (y + z) == x * y + x * From;
	// z right
	// (x + y) * z == x * y + x * z;
    }
};


concept CommutativeRing<typename Element>
  : Ring<Element>,
    MultiplicativeCommutativeSemiGroup<Element>
{};


concept RingWithIdentity<typename Element>
  : Ring<Element>,
    MultiplicativeMonoid<Element>
{};
 

concept CommutativeRingWithIdentity<typename Element>
  : Ring<Element>,
    MultiplicativeCommutativeMonoid<Element>
{};
 

concept DivisionRing<typename Element>
  : RingWithIdentity<Element>,
    DividableWithAssign<Element>
{
    axiom Consistency(Element x, Element y)
    {
	// I don't know how to express consistency between / and /=
    }  

    axiom ZeroIsDifferentFromOne()
    {
	// 0 != 1
	// Note that it is possible to allow 0 == 1 in a DivisionRing, this structure would be even
	// a field. On the other hand this Field would only contain one single element
	// as a consequence of 0 == 1. It is called the trivial field and of no practical value.
	// Therefore, we exclude this field and require 0 != 1.
 
	identity<math::add<Element>, Element> != identity<math::mult<Element>, Element>;
    }

    axiom NonZeroDividability(Element x)
    {
	// if (x != 0) x / x == 1
	if (x != identity<math::add<Element>, Element>) 
	    x / x == identity<math::mult<Element>, Element>;
    }
};    


concept Field<typename Element>
  : DivisionRing<Element>,
    MultiplicativeCommutativeMonoid<Element>
{};



// ====================================
// Miscellaneous concepts
// that shall find a better place later
// ====================================


// Closure of EqualityComparable under a binary operation:
// That is, the result of this binary operation is also EqualityComparable
// with itself and with the operand type.
auto concept Closed2EqualityComparable<typename Operation, typename Element>
  : BinaryIsoFunction<Operation, Element>
{
    where std::EqualityComparable<Element>;
    where std::EqualityComparable< BinaryIsoFunction<Operation, Element>::result_type >;
    where std::EqualityComparable< Element, BinaryIsoFunction<Operation, Element>::result_type >;
    where std::EqualityComparable< BinaryIsoFunction<Operation, Element>::result_type, Element >;
};


// LessThanComparable will have the other operators when defaults are supported
// At this point the following isn't needed anymore
auto concept FullLessThanComparable<typename T, typename U = T>
{
    bool operator<(const T&, const U&);
    bool operator<=(const T&, const U&);
    bool operator>(const T&, const U&);
    bool operator>=(const T&, const U&);
};


// Same for LessThanComparable
auto concept Closed2LessThanComparable<typename Operation, typename Element>
  : BinaryIsoFunction<Operation, Element>
{
    where FullLessThanComparable<Element>;
    where FullLessThanComparable< BinaryIsoFunction<Operation, Element>::result_type >;
    where FullLessThanComparable< Element, BinaryIsoFunction<Operation, Element>::result_type >;
    where FullLessThanComparable< BinaryIsoFunction<Operation, Element>::result_type, Element >;
};


} // namespace math


#endif // LA_WITH_CONCEPTS

#endif // LA_CONCEPTS_INCLUDE
