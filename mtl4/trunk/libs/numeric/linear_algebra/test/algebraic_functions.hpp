#ifndef algebraic_functions_include
#define algebraic_functions_include

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <boost/config/concept_macros.hpp> 

#ifdef LA_WITH_CONCEPTS
#  include <bits/concepts.h>
#endif

#include <boost/numeric/linear_algebra/concepts.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>

// Pure algebraic functions (mainly for applying concepts)

namespace mtl {


// {Op, Element} must be a Magma
// The result of the operation must be EqualityComparable
// Closed2EqualityComparable requires the comparability of any combination of
// Element and result_type.
// It is thus more limiting than necessary, well slightly.
template <typename Op, typename Element>
#if 0
  LA_WHERE( math::Magma<Op, Element>
	    && std::EqualityComparable< math::Magma<Op, Element>::result_type > )
#endif
  LA_WHERE( math::Closed2EqualityComparable<Op, Element> && math::Magma<Op, Element> )
inline bool equal_results(const Element& v1a, const Element& v1b, 
			  const Element& v2a, const Element& v2b, Op op) 
{
#if 0
    // If we use EqualityComparable for Element we cannot compare the results
    // directly since Magma only assumes that the results are convertible to Element
    // which does not imply that they are comparable.
    // Assigning to temporaries will the conversion and the temporaries are comparable.

    Element res1= op(v1a, v1b), res2= op(v2a, v2b);
    return res1 == res2;
#endif

    return op(v1a, v1b) == op(v2a, v2b);
}


// Same for AdditiveMagma
template <typename Element>
LA_WHERE ( math::AdditiveMagma<Element> 
	   && math::Closed2EqualityComparable< math::add<Element>, Element > )
inline bool equal_add_results(const Element& v1a, const Element& v1b, 
			      const Element& v2a, const Element& v2b) 
{
    return v1a + v1b == v2a + v2b;
}





// {Op, Element} must be a Monoid
// The result of the operation must be EqualityComparable
template <typename Op, typename Element>
  LA_WHERE( math::Closed2EqualityComparable<Op, Element> && math::Monoid<Op, Element> )
inline bool identity_pair(const Element& v1, const Element& v2, Op op) 
{
    using math::identity;
    return op(v1, v2) == identity<Op, Element>()(v1) ;
}


// {Op, Element} must be a Monoid
template <typename Op, typename Element, typename Exponent>
  LA_WHERE( math::Monoid<Op, Element> 
            && std::Integral<Exponent> )             // TBD: not minimal requirements, will refine it later
inline Element multiply_and_square(Element base, Exponent exp, Op op) 
{
    using math::identity;
    Element value= identity<Op, Element>()(base), square= base;
    for (; exp > 0; exp>>= 1) {
	if (exp & 1) 
	    value= op(value, square);
	square= op(square, square); 
    }
    return value;  
} 


// {Op, Element} must be a Group
// Element must be LessThanComparable
// Under construction w.r.t. semantic requirements, introduction of ordered group needed
template <typename Op, typename Element>
  LA_WHERE( math::Closed2LessThanComparable<Op, Element> 
	    && math::Group<Op, Element> )
inline int algebraic_division(const Element& v1, const Element& v2, Op op)
{
    using math::identity; using math::inverse;

    // Temporaries to avoid redundant operations
    Element id= identity<Op, Element>()(v1),     // Identity
            iv2= inverse<Op, Element>()(v2),     // Inverse of v2
   	    tmp(v1);                             // Copy of v1, will be lessened until < id
    
    if (v1 <= id) return 0;
    int counter= 0;
    for (; tmp > id; counter++) 
	tmp= op(tmp, iv2);
    // counter only correct if tmp == id
    if (tmp < id) 
	counter--;
    return counter;
}




#if 0

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
