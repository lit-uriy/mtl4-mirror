// $COPYRIGHT$

#ifndef LA_ALGEBRAIC_CONCEPTS_INCLUDE
#define LA_ALGEBRAIC_CONCEPTS_INCLUDE

#ifndef __GXX_CONCEPTS__
#  warning "Concepts are not used"
#else

#  include <concepts>

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>

namespace algebra {

concept Commutative<typename Operation, typename Element>
{
    axiom Commutativity(Operation op, Element x, Element y)
    {
	op(x, y) == op(y, x); 
    }   
};


concept Associative<typename Operation, typename Element>
{
    axiom Associativity(Operation op, Element x, Element y, Element z)
    {
	op(x, op(y, z)) == op(op(x, y), z); 
    }
};
    

auto concept SemiGroup<typename Operation, typename Element>
  : Associative<Operation, Element>
{};


concept Monoid<typename Operation, typename Element>
  : SemiGroup<Operation, Element> 
{
    typename identity_result_type;
    identity_result_type identity(Operation, Element);

    axiom Neutrality(Operation op, Element x)
    {
	op( x, identity(op, x) ) == x;
	op( identity(op, x), x ) == x;
    }
};


auto concept Inversion<typename Operation, typename Element>
{
    typename inverse_result_type;
    inverse_result_type inverse(Operation, Element);
 
};


concept Group<typename Operation, typename Element>
  : Monoid<Operation, Element>, Inversion<Operation, Element>
{
    axiom Inversion(Operation op, Element x)
    {
	op( x, inverse(op, x) ) == identity(op, x);
	op( inverse(op, x), x ) == identity(op, x);
    }
};


auto concept AbelianGroup<typename Operation, typename Element>
  : Group<Operation, Element>, Commutative<Operation, Element>
{};


concept Distributive<typename AddOp, typename MultOp, typename Element>
{
    axiom Distributivity(AddOp add, MultOp mult, Element x, Element y, Element z)
    {
	// From left
	mult(x, add(y, z)) == add(mult(x, y), mult(x, z));
	// z right
	mult(add(x, y), z) == add(mult(x, z), mult(y, z));
    }
};


auto concept Ring<typename AddOp, typename MultOp, typename Element>
  : AbelianGroup<AddOp, Element>,
    SemiGroup<MultOp, Element>,
    Distributive<AddOp, MultOp, Element>
{};


auto concept RingWithIdentity<typename AddOp, typename MultOp, typename Element>
  : Ring<AddOp, MultOp, Element>,
    Monoid<MultOp, Element>
{};


// Also called SkewField
concept DivisionRing<typename AddOp, typename MultOp, typename Element>
  : RingWithIdentity<AddOp, MultOp, Element>,
    Inversion<MultOp, Element>
{
    // 0 != 1, otherwise trivial
    axiom ZeroIsDifferentFromOne(AddOp add, MultOp mult, Element x)
    {
	identity(add, x) != identity(mult, x);       
    }

    // Non-zero divisibility from left and from right
    axiom NonZeroDivisibility(AddOp add, MultOp mult, Element x)
    {
	if (x != identity(add, x))
	    mult(inverse(mult, x), x) == identity(mult, x);
	if (x != identity(add, x))
	    mult(x, inverse(mult, x)) == identity(mult, x);
    }
};    


auto concept Field<typename AddOp, typename MultOp, typename Element>
  : DivisionRing<AddOp, MultOp, Element>,
    Commutative<MultOp, Element>
{};


} // namespace algebra

#endif  // __GXX_CONCEPTS__

#endif // LA_ALGEBRAIC_CONCEPTS_INCLUDE
