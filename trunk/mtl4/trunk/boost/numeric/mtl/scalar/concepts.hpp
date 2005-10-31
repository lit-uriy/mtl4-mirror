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



} // namespace mtl

#endif // MTL_CONCEPTS_INCLUDE
