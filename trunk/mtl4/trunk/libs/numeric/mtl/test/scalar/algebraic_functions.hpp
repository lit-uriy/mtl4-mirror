#ifndef algebraic_functions_include
#define algebraic_functions_include

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

// #include "enable_if.hpp"
// #include "property_traits.hpp"
// #include <glas/glas.hpp>

#include <bits/concepts.h>
#include <boost/numeric/mtl/scalar/concepts.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

// Pure algebraic functions (mainly for applying concepts)

namespace mtl {


// {T, Op} must be a Magma
// T must be EqualityComparable
template <class T, class Op> 
#if 0
  where math::Magma<T, Op>
        && std::EqualityComparable< math::Magma<T, Op>::result_type > 
#endif
  where std::EqualityComparable<T> && math::Magma<T, Op>
inline bool equal_results(const T& v1a, const T& v1b, 
			  const T& v2a, const T& v2b, Op op) 
{
    T res1= op(v1a, v1b), res2= op(v2a, v2b);
    return res1 == res2;

#if 0
    return op(v1a, v1b) == op(v2a, v2b);
#endif
}

// {T, Op} must be a Monoid
// T must be EqualityComparable
template <class T, class Op>
  where std::EqualityComparable<T> && math::Monoid<T, Op>
inline bool identity_pair(const T& v1, const T& v2, Op op) 
{
    using math::identity;
    T res1= op(v1, v2), res2= identity<T, Op>()(v1);
    return res1 == res2;

    // return op(v1, v2) == math::identity<T, Op>()(v1) ;
}

// {T, Op} must be a Monoid
template <class T, class Op>
  where std::CopyConstructible<T> && math::Monoid<T, Op>
inline T multiplyAndSquare(T base, int exp, Op op) 
{
    T value= math::identity<T, Op>()(base), square= base;
    for (; exp > 0; exp>>= 1) {
	if (exp & 1) value= op(value, square);
	square= op(square, square); 
    }
    return value;  
} 

#if 0

// {T, Op} must be a Group
// T must be LessThanComparable and Assignable
template <class T, class Op>
inline int poorMensDivision(const T& v1, const T& v2, Op op) {
  // copies to avoid redundant operations
  T id= glas::identity<T, Op>()(), iv2= glas::inverse<T, Op>()(v2), tmp(v1);

  if (v1 <= id) return 0;
  int counter= 0;
  for (; tmp > id; counter++) tmp= op(tmp, iv2);
  if (tmp < id) counter--;
  return counter;
}

// {T, Op} must be a Group
// T must be LessThanComparable and Assignable
template <class T, class Op>
inline int poorMensAbsDivision(const T& v1, const T& v2, Op op) {
  // copies to avoid redundant operations
  T id= glas::identity<T, Op>()(), iv2(v2 < id ? v2 : glas::inverse<T, Op>()(v2)),
    va1(v1 < id ? glas::inverse<T, Op>()(v1) : v1), tmp(va1);
  if (va1 <= id) return 0;
  int counter= 0;
  for (; tmp > id; counter++) tmp= op(tmp, iv2);
  if (tmp < id) counter--;
  return counter;
}


// {Iter*, Op} must be a CommutativeMonoid
struct sortedAccumulate_t {
  template <class Iter, class Op, class Comp>
  typename enable_if<glas::is_associative<typename std::iterator_traits<Iter>::value_type, Op>::value 
                       && glas::is_commutative<typename std::iterator_traits<Iter>::value_type, Op>::value, 
		     typename std::iterator_traits<Iter>::value_type>::type
  operator() (Iter first, Iter last, Op op, Comp comp) {
    std::cout << "sortedAccumulate_t\n";
    typedef typename std::iterator_traits<Iter>::value_type value_type;
    std::vector<value_type> tmp(first, last);
    std::sort(tmp.begin(), tmp.end(), comp); 
    return std::accumulate(tmp.begin(), tmp.end(), glas::identity<value_type, Op>()(), op); }
} sortedAccumulate;

// {Iter*, Op} must be a Monoid
struct unsortedAccumulate_t {
  template <class Iter, class Op>
  typename std::iterator_traits<Iter>::value_type
  operator() (Iter first, Iter last, Op op) {
    std::cout << "unsortedAccumulate_t\n";
    typedef typename std::iterator_traits<Iter>::value_type value_type;
    return std::accumulate(first, last, glas::identity<value_type, Op>()(), op); }

  // Only for Compability
  template <class Iter, class Op, class Comp>
  typename std::iterator_traits<Iter>::value_type
  operator() (Iter first, Iter last, Op op, Comp) {
    return operator() (first, last, op); }
} unsortedAccumulate;
    
// {Iter*, Op} must be a Monoid
template <class Iter, class Op, class Comp>
inline typename std::iterator_traits<Iter>::value_type
trySortedAccumulate(Iter first, Iter last, Op op, Comp comp) {
  typedef typename std::iterator_traits<Iter>::value_type value_type;
  typename if_type<glas::is_associative<value_type, Op>::value && glas::is_commutative<value_type, Op>::value,  
                   sortedAccumulate_t, unsortedAccumulate_t>::type  accumulate;
  // alternatively checking for structure
  //   typename if_type<glas::is_commutative_monoid<value_type, Op>::value,  
  //                    sortedAccumulate_t, unsortedAccumulate_t>::type  accumulate;
  return accumulate(first, last, op, comp);
}

#endif // 0

} // namespace mtl

#endif // algebraic_functions_include
