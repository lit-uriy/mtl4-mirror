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
template <typename Set, typename Operation>
struct concept Magma
  : std::Callable2<Operation, Set, Set>
{
    where std::Assignable<Set>;

    // Short version
    where result_type == Set;
    
    // Long version if multiple refined concepts have result_type
    // where std::Callable2<Operation, Set, Set>::result_type == Set;
};
#endif

template <typename Set, typename Operation>
struct concept Magma
{
    typename result_type = Set;
    Set operator() (Operation, Set, Set);
    Set& operator= (Set&, Set const&);
};

// SemiGroup is a refinement which must be nominal
template <typename Set, typename Operation>
concept SemiGroup
  : Magma<Set, Operation>
{};


template <typename Set, typename Operation>
concept CommutativeSemiGroup
  : SemiGroup<Set, Operation>
{};

// Adding identity
template <typename Set, typename Operation>
concept Monoid
: SemiGroup<Set, Operation> 
{
    Set operator() (glas::identity<Set, Operation>);
};

template <typename Set, typename Operation>
concept CommutativeMonoid
  : Monoid<Set, Operation>
{};


template <typename Set, typename Operation>
concept PartiallyInvertibleMonoid
  : Monoid<Set, Operation> 
{
    bool operator() (glas::is_invertible<Set, Operation>, Set);
};

template <typename Set, typename Operation>
concept PartiallyInvertibleCommutativeMonoid
  : PartiallyInvertibleMonoid<Set, Operation>, CommutativeMonoid<Set, Operation>
{};

template <typename Set, typename Operation>
concept Group
  : PartiallyInvertibleMonoid<Set, Operation>
{
    Set operator() (glas::inverse<Set, Operation>, Set);
};

template <typename Set, typename Operation>
concept AbelianGroup
  : Group<Set, Operation>, PartiallyInvertibleCommutativeMonoid<Set, Operation>
{};


#endif // __GXX_CONCEPTS__

} // namespace mtl

#endif // MTL_CONCEPTS_INCLUDE
