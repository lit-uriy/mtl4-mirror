// $COPYRIGHT$

#ifndef LA_CONCEPTS_INCLUDE
#define LA_CONCEPTS_INCLUDE

#include <boost/config/concept_macros.hpp>

#ifdef LA_WITH_CONCEPTS
#  include <concepts>
#else
#  warning "Concepts are not used"
#endif


#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/linear_algebra/operators.hpp>

// If desired one can disable the default concept maps with LA_NO_CONCEPT_MAPS


#include <complex>


namespace math {

#ifdef LA_WITH_CONCEPTS



// ================
// Utility Concepts
// ================

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




// ==================
// Algebraic Concepts
// ==================


auto concept Magma<typename Operation, typename Element>
    : BinaryIsoFunction<Operation, Element>
{
    where std::Assignable<Element>;
    where std::Assignable<Element, BinaryIsoFunction<Operation, Element>::result_type>;
};


// For algebraic structures that are commutative but not associative
// As an example floating point numbers are commutative but not associative
//   w.r.t. addition and multiplication
concept CommutativeMagma<typename Operation, typename Element>
  : Magma<Operation, Element>
{
    axiom Commutativity(Operation op, Element x, Element y)
    {
	op(x, y) == op(y, x); 
    }   
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
  : SemiGroup<Operation, Element>,
    CommutativeMagma<Operation, Element>
{};


// Adding identity
concept Monoid<typename Operation, typename Element>
  : SemiGroup<Operation, Element> 
{
    where UnaryIsoFunction< identity<Operation, Element>, Element >;

    axiom Neutrality(Operation op, Element x)
    {
	op( x, identity<Operation, Element>()(x) ) == x;
	op( identity<Operation, Element>()(x), x ) == x;
    }
};


concept CommutativeMonoid<typename Operation, typename Element>
  : CommutativeSemiGroup<Operation, Element>, 
    Monoid<Operation, Element>
{};


concept PartiallyInvertibleMonoid<typename Operation, typename Element>
  : Monoid<Operation, Element> 
{
    where std::Predicate< math::is_invertible<Operation, Element>, Element >;

    where UnaryIsoFunction< inverse<Operation, Element>, Element >; 

    axiom Inversion(Operation op, Element x)
    {
	// Only for invertible elements:
	// if ( is_invertible<Operation, Element>()(x) )
	//     op( x, inverse<Operation, Element>()(x) ) == identity<Operation, Element>()(x);   NYS
	// if ( is_invertible<Operation, Element>()(x) )
	//     op( inverse<Operation, Element>()(x), x ) == identity<Operation, Element>()(x);   NYS
    }
};


concept PartiallyInvertibleCommutativeMonoid<typename Operation, typename Element>
  : PartiallyInvertibleMonoid<Operation, Element>, 
    CommutativeMonoid<Operation, Element>   
{};


concept Group<typename Operation, typename Element>
  : PartiallyInvertibleMonoid<Operation, Element>
{
    axiom AlwaysInvertible(Operation op, Element x)
    {
	is_invertible<Operation, Element>()(x);
    }

    axiom Inversion(Operation op, Element x)
    {
	// In fact this is implied by AlwaysInvertible and inherited Inversion axiom
	// However, we don't rely on the compiler to deduce this
	op( x, inverse<Operation, Element>()(x) ) == identity<Operation, Element>()(x);
	op( inverse<Operation, Element>()(x), x ) == identity<Operation, Element>()(x);
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
{
    where Magma< math::add<Element>, Element >;

    typename assign_result_type;  
    assign_result_type operator+=(Element& x, Element y);

    // Operator + is by default defined with +=
    typename result_type;  
    result_type operator+(Element x, Element y);
#if 0
    {
	Element tmp(x);
	return tmp += y;                      defaults NYS
    }
#endif 
    
    // Type consistency with Magma
    where std::SameType< result_type, 
	                 Magma< math::add<Element>, Element >::result_type>;

    axiom Consistency(math::add<Element> op, Element x, Element y)
    {
	op(x, y) == x + y;                    
	// Might change later
        x + y == x += y;
	// Element tmp = x; tmp+= y; tmp == x + y; not proposal-compliant
    }   
}


concept AdditiveCommutativeMagma<typename Element>
  : AdditiveMagma<Element> 
{
    where CommutativeMagma< math::add<Element>, Element >;
};


concept AdditiveSemiGroup<typename Element>
  : AdditiveMagma<Element>
{
    where SemiGroup< math::add<Element>, Element >;
};


// We really need only one of the additive concepts for the requirements, 
// the requirements of the other would be implied.
// Vice versa, to derive concept maps of nested concepts from
// concept maps of refined concepts, they are needed all.
concept AdditiveCommutativeSemiGroup<typename Element>
  : AdditiveSemiGroup<Element>,
    AdditiveCommutativeMagma<Element>
{
    where CommutativeSemiGroup< math::add<Element>, Element >;
};


concept AdditiveMonoid<typename Element>
  : AdditiveSemiGroup<Element>
{
    where Monoid< math::add<Element>, Element >;
};


// We really need only one of the additive concepts for the requirements, 
// the requirements of the other would be implied.
// Vice versa, to derive concept maps of nested concepts from
// concept maps of refined concepts, they are needed all.
concept AdditiveCommutativeMonoid<typename Element>
  : AdditiveMonoid<Element>,
    AdditiveCommutativeSemiGroup<Element>
{
    where CommutativeMonoid< math::add<Element>, Element >;
};


concept AdditivePartiallyInvertibleMonoid<typename Element>
  : AdditiveMonoid<Element>
{
    where PartiallyInvertibleMonoid< math::add<Element>, Element >;

    typename assign_result_type;  
    assign_result_type operator-=(Element& x, Element y);
     
    // Operator - by default defined with -=
    typename result_type;  
    result_type operator-(Element& x, Element y);
#if 0
    {
	Element tmp(x);
	return tmp -= y;                      defaults NYS
    }
#endif 

    // #if 0 // end of input localization

    typename unary_result_type;  
    unary_result_type operator-(Element x);
#if 0
    {
	return identity<math::add<Element>, Element>()(x) - x;      defaults NYS
    }
#endif 
    
    axiom Consistency(math::add<Element> op, Element x, Element y)
    {
#if 0
	// Conditional axioms have problems
	// -->  error: expected `(' before 'if'   '
	// consistency between additive and pure algebraic concept
	if ( is_invertible<math::add<Element>, Element>()(y) )
	    op(x, inverse<math::add<Element>, Element>() (y)) == x - y;            
	if ( is_invertible<math::add<Element>, Element>()(y) )
	    inverse<math::add<Element>, Element>() (y) == -y;                      

	// consistency between unary and binary -
	if ( is_invertible<math::add<Element>, Element>()(x) )
	    identity<math::add<Element>, Element>() (x) - x == -x;                 

	// Might change later
	if ( is_invertible<math::add<Element>, Element>()(y) )
	    x - y == x -= y;                                                       
	// Element tmp = x; tmp-= y; tmp == x - y; not proposal-compliant
#endif 
    }  

};


concept AdditivePartiallyInvertibleCommutativeMonoid<typename Element>
  : AdditivePartiallyInvertibleMonoid<Element>,
    AdditiveCommutativeMonoid<Element>
{
    where PartiallyInvertibleCommutativeMonoid< math::add<Element>, Element >;
};



concept AdditiveGroup<typename Element>
  : AdditivePartiallyInvertibleMonoid<Element>
{
    where Group< math::add<Element>, Element >;
};


concept AdditiveAbelianGroup<typename Element>
  : AdditiveGroup<Element>,
    AdditiveCommutativeMonoid<Element>
{
    where AbelianGroup< math::add<Element>, Element >;
};


// ============================
// Multiplitive scalar concepts
// ============================


auto concept MultiplicativeMagma<typename Element>
{
    where Magma< math::mult<Element>, Element >;

    typename assign_result_type;  
    assign_result_type operator*=(Element& x, Element y);

    // Operator * is by default defined with *=
    typename result_type;  
    result_type operator*(Element x, Element y);
#if 0
    {
	Element tmp(x);
	return tmp *= y;                      defaults NYS
    }
#endif 
    
    // Type consistency with Magma
    where std::SameType< result_type, 
	                 Magma< math::mult<Element>, Element >::result_type>;

    axiom Consistency(math::mult<Element> op, Element x, Element y)
    {
	op(x, y) == x * y;                    
	// Might change later
        x * y == x *= y;
	// Element tmp = x; tmp*= y; tmp == x * y; not proposal-compliant
    }  

}


concept MultiplicativeSemiGroup<typename Element>
  : MultiplicativeMagma<Element>
{ 
    where SemiGroup< math::mult<Element>, Element >;
};


concept MultiplicativeCommutativeSemiGroup<typename Element>
  : MultiplicativeSemiGroup<Element>
{
    where CommutativeSemiGroup< math::mult<Element>, Element >;
};


concept MultiplicativeMonoid<typename Element>
  : MultiplicativeSemiGroup<Element>
{
    where Monoid< math::mult<Element>, Element >;
};


concept MultiplicativeCommutativeMonoid<typename Element>
  : MultiplicativeMonoid<Element>,
    MultiplicativeCommutativeSemiGroup<Element>
{
    where CommutativeMonoid< math::mult<Element>, Element >;
};


concept MultiplicativePartiallyInvertibleMonoid<typename Element>
  : MultiplicativeMonoid<Element>
{
    where PartiallyInvertibleMonoid< math::mult<Element>, Element >;

    typename assign_result_type;  
    assign_result_type operator/=(Element& x, Element y);
     
    // Operator / by default defined with /=
    typename result_type;  
    result_type operator/(Element& x, Element y);
#if 0
    {
	Element tmp(x);
	return tmp /= y;                      defaults NYS
    }
#endif 
    
    axiom Consistency(math::mult<Element> op, Element x, Element y)
    {
#if 0
	// consistency between multiplicative and pure algebraic concept
	if ( is_invertible<math::mult<Element>, Element>()(y) )
	    op(x, inverse<math::mult<Element>, Element>() (y)) == x / y;           NYS

	// Consistency between / and /=, might change later
	if ( is_invertible<math::mult<Element>, Element>()(y) )
	    x / y == x /= y;                                                       NYS
	// Element tmp = x; tmp/= y; tmp == x / y; not proposal-compliant
#endif 
    }  
};


concept MultiplicativePartiallyInvertibleCommutativeMonoid<typename Element>
  : MultiplicativePartiallyInvertibleMonoid<Element>,
    MultiplicativeCommutativeMonoid<Element>
{
    where PartiallyInvertibleCommutativeMonoid< math::mult<Element>, Element >;
};


concept MultiplicativeGroup<typename Element>
  : MultiplicativeMonoid<Element>
{        
    where Group< math::mult<Element>, Element >;
};


concept MultiplicativeAbelianGroup<typename Element>
  : MultiplicativeGroup<Element>,
    MultiplicativeCommutativeMonoid<Element>
{
    where AbelianGroup< math::mult<Element>, Element >;
};


// ======================================
// Algebraic concepts with two connectors
// ======================================

// ------------------
// Based on operators 
// ------------------

// Handier, less generic


// Alternative definitions use MultiplicativeMonoid<Element> for Ring
// and call such concepts Pseudo-Ring

concept Ring<typename Element>
  : AdditiveAbelianGroup<Element>,
    MultiplicativeSemiGroup<Element>
{
    axiom Distributivity(Element x, Element y, Element z)
    {
	// From left
	x * (y + z) == x * y + x * z;
	// From right
	(x + y) * z == x * y + x * z;
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
  : RingWithIdentity<Element>,
    CommutativeRing<Element>
{};
 

concept DivisionRing<typename Element>
  : RingWithIdentity<Element>,
    MultiplicativePartiallyInvertibleMonoid<Element>
{
    axiom Consistency(Element x, Element y)
    {
	// I don't know how to express consistency between / and /=
    }  

    axiom ZeroIsDifferentFromOne(Element x)
    {
	// 0 != 1
	// Note that it is possible to allow 0 == 1 in a DivisionRing, this structure would be even
	// a field. On the other hand this Field would only contain one single element
	// as a consequence of 0 == 1. It is called the trivial field and of no practical value.
	// Therefore, we exclude this field and require 0 != 1.
 
	identity<math::add<Element>, Element>()(x) != identity<math::mult<Element>, Element>()(x);
    }

    axiom NonZeroDivisibility(Element x)
    {
	// if (x != 0) x / x == 1
	// if (x != identity<math::add<Element>, Element>()(x))                                         NYS
	//     x / x == identity<math::mult<Element>, Element>()(x);
    }
};    


concept Field<typename Element>
  : DivisionRing<Element>,
    CommutativeRingWithIdentity<Element>
{};

// -----------------
// Based on functors
// -----------------

// More generic, less handy to use

concept GenericRing<typename AddOp, typename MultOp, typename Element>
{
    where AbelianGroup<AddOp, Element>;

    where SemiGroup<MultOp, Element>;

    axiom Distributivity(AddOp add, MultOp mult, Element x, Element y, Element z)
    {
	// From left
	// mult(x, add(y, z)) == add(mult(x, y), mult(x, z));
	// z right
	// mult(add(x, y), z) == add(mult(x, z), mult(y, z));
    }
};


concept GenericCommutativeRing<typename AddOp, typename MultOp, typename Element>
  : GenericRing<AddOp, MultOp, Element>
{
    where CommutativeSemiGroup<MultOp, Element>;
};


concept GenericRingWithIdentity<typename AddOp, typename MultOp, typename Element>
  : GenericRing<AddOp, MultOp, Element>
{
    where Monoid<MultOp, Element>;
};


concept GenericCommutativeRingWithIdentity<typename AddOp, typename MultOp, typename Element>
  : GenericRingWithIdentity<AddOp, MultOp, Element>,
    GenericCommutativeRing<AddOp, MultOp, Element>
{};


concept GenericDivisionRing<typename AddOp, typename MultOp, typename Element>
  : GenericRingWithIdentity<AddOp, MultOp, Element>
{
    where UnaryIsoFunction< inverse<MultOp, Element>, Element >; 

    axiom ZeroIsDifferentFromOne(Element x)
    {
	// 0 != 1
	// Comments see DivisionRing

	// identity<AddOp, Element>()(x) != identity<MultOp, Element>()(x);        NYS
    }

    axiom NonZeroDivisibility(MultOp mult, Element x)
    {
	// if (x != 0) x / x == 1
	// if (x != identity<AddOp, Element>()(x))                                         NYS
	//     mult(x, inverse<MultOp, Element>()(x)) == identity<MultOp, Element>()(x);
    }
};    


concept GenericField<typename AddOp, typename MultOp, typename Element>
  : GenericDivisionRing<AddOp, MultOp, Element>,
    GenericCommutativeRingWithIdentity<AddOp, MultOp, Element>
{};


// ======================
// Miscellaneous concepts
// ======================

// that shall find a better place later


// EqualityComparable will have the != when defaults are supported
// At this point the following won't needed anymore
auto concept FullEqualityComparable<typename T, typename U = T>
{
  //where std::EqualityComparable<T, U>;

    bool operator==(const T&, const U&);
    bool operator!=(const T&, const U&);
};

// Closure of EqualityComparable under a binary operation:
// That is, the result of this binary operation is also EqualityComparable
// with itself and with the operand type.
auto concept Closed2EqualityComparable<typename Operation, typename Element>
  : BinaryIsoFunction<Operation, Element>
{
    where FullEqualityComparable<Element>;
    where FullEqualityComparable< BinaryIsoFunction<Operation, Element>::result_type >;
    where FullEqualityComparable< Element, BinaryIsoFunction<Operation, Element>::result_type >;
    where FullEqualityComparable< BinaryIsoFunction<Operation, Element>::result_type, Element >;
};


// LessThanComparable will have the other operators when defaults are supported
// At this point the following won't needed anymore
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

#if 0
auto concept NumericOperatorResultConvertible<typename T>
  : AddableWithAssign<T>,
    SubtractableWithAssign<T>,
    MultiplicableWithAssign<T>,
    DivisibleWithAssign<T>
{
    where std::Convertible< AddableWithAssign<T>::result_type, T>;
    where std::Convertible< SubtractableWithAssign<T>::result_type, T>;
    where std::Convertible< MultiplicableWithAssign<T>::result_type, T>;
    where std::Convertible< DivisibleWithAssign<T>::result_type, T>;
}
#endif

auto concept AdditionResultConvertible<typename T>
{
    typename result_type;
    result_type operator+(T t, T u);
    where std::Convertible<result_type, T>;

    typename result_type;
    result_type operator+=(T& t, T u);
    where std::Convertible<result_type, T>;
};    


auto concept SubtractionResultConvertible<typename T>
{
    typename result_type;
    result_type operator-(T t, T u);
    where std::Convertible<result_type, T>;

    typename result_type;
    result_type operator-=(T& t, T u);
    where std::Convertible<result_type, T>;
};    

auto concept NumericOperatorResultConvertible<typename T>
  : AdditionResultConvertible<T>,
    SubtractionResultConvertible<T>
{};

// ====================
// Default Concept Maps
// ====================

#ifndef LA_NO_CONCEPT_MAPS

// ==============
// Integral Types
// ==============

// concept_map AbelianGroup<math::add<char>, char> {}
// concept_map AbelianGroup<math::add<short>, short> {}
concept_map AbelianGroup<math::add<int>, int> {}
concept_map AbelianGroup<math::add<long>, long> {}
concept_map AbelianGroup<math::add<long long>, long long> {}

// concept_map CommutativeMonoid<math::mult<char>, char> {}
// concept_map CommutativeMonoid<math::mult<short>, short> {}
concept_map CommutativeMonoid<math::mult<int>, int> {}
concept_map CommutativeMonoid<math::mult<long>, long> {}
concept_map CommutativeMonoid<math::mult<long long>, long long> {}

// concept_map CommutativeRingWithIdentity<char> {}
// concept_map CommutativeRingWithIdentity<short> {}
concept_map CommutativeRingWithIdentity<int> {}
concept_map CommutativeRingWithIdentity<long> {}
concept_map CommutativeRingWithIdentity<long long> {}


// ====================
// Floating Point Types
// ====================

concept_map AbelianGroup<math::add<float>, float> {}
concept_map AbelianGroup<math::add<double>, double> {}

concept_map PartiallyInvertibleCommutativeMonoid<math::mult<float>, float> {} 
concept_map PartiallyInvertibleCommutativeMonoid<math::mult<double>, double> {} 

concept_map Field<float> {}
concept_map Field<double> {}

#if 0
    // Convertibility from results into complex<T> must be expressed properly for co
    template <typename T>
      where Field<T> && NumericOperatorResultConvertible< std::complex<T> >
    concept_map Field< std::complex<T> > {}
#endif

concept_map AbelianGroup< math::add< std::complex<float> >, std::complex<float> > {}
concept_map AbelianGroup< math::add< std::complex<double> >, std::complex<double> > {}

concept_map PartiallyInvertibleCommutativeMonoid<math::mult< std::complex<float> >, std::complex<float> > {} 
concept_map PartiallyInvertibleCommutativeMonoid<math::mult< std::complex<double> >, std::complex<double> > {}

concept_map Field< std::complex<float> > {}
concept_map Field< std::complex<double> > {}

#endif // LA_NO_CONCEPT_MAPS

#endif // LA_WITH_CONCEPTS


// =================================================
// Concept to specify return type of abs (and norms)
// =================================================


#ifdef LA_WITH_CONCEPTS

// Concept to specify return type of abs and norms
concept ScalarWithMagnitude<typename T>
{
    typename type;
};

template <typename T>
  where std::Integral<T>
concept_map ScalarWithMagnitude<T>
{
    typedef T type;
}

concept_map ScalarWithMagnitude<float> { typedef float magnitude_type; };
concept_map ScalarWithMagnitude<double> { typedef double magnitude_type; };

template <typename T>
  where ScalarWithMagnitude<T>
concept_map ScalarWithMagnitude< std::complex<T> >
{
    typedef T type;
    // Or recursively ?
    // typedef ScalarWithMagnitude<T>::magnitude_type type;
};


// concept_map ScalarWithMagnitude<short> { typedef short magnitude_type };

#else

// For the moment everything is its own magnitude type, unless stated otherwise
template <typename T>
struct magnitude_type_trait
{
    typedef T type;
};

template <typename T>
struct magnitude_type_trait< std::complex<T> >
{
    typedef T type;
};

#endif




} // namespace math



#endif // LA_CONCEPTS_INCLUDE
