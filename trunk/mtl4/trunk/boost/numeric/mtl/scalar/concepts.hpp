// $COPYRIGHT$

#ifndef MTL_CONCEPTS_INCLUDE
#define MTL_CONCEPTS_INCLUDE

#include <bits/concepts.h>

namespace mtl {

#if 0
template <typename Operation, typename Operand1, typename Operand2 = Operand1>
struct concept BinaryOperation
{
    typedef result_type;

    result_type operator() (Operand1, Operand2);
};

template<typename T>
struct concept Assignable
{
  T& operator=(T& x, const T& y);
};

template <typename Set, typename Operation>
struct concept Magma
  : Assignable<Set>
{
    Set Operation::operator() (Set, Set);
};
    
template <typename Set, typename Operation>
struct concept Magma
  : std::Assignable<Set>, std::
    
template <typename Set, typename Operation>
struct concept Magma
  : std::Assignable<Set>, Callable2<Operation, Set, Set>
{
    typename result_type = Set;
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
