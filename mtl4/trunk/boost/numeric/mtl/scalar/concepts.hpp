// $COPYRIGHT$

#ifndef MTL_CONCEPTS_INCLUDE
#define MTL_CONCEPTS_INCLUDE

#include <bits/concepts.h>

namespace mtl {
    
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

// SemiGroup is a refinement which must be nominal
template <typename Set, typename Operation>
concept CommutativeSemiGroup
  : SemiGroup<Set, Operation>
{};

#if 0
// Just a mapping from one set to the same
template <typename Set, typename Operation>
struct concept EndomorphicConstant
{
    Set operator();
};
#endif

template <typename Set, typename Operation>
concept Monoid
  : SemiGroup<Set, Operation>
{
    typename identity = glas::identity<Set, Operation>;
    // where EndomorphicConstant< identity<Set, Operation> >;
};

} // namespace mtl

#endif // MTL_CONCEPTS_INCLUDE
