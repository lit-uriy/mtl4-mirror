#ifndef MTL_ACCUMULATION_TEST_INCLUDE
#define MTL_ACCUMULATION_TEST_INCLUDE


#include <iostream>
#include <algorithm>
#include <concepts>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/new_concepts.hpp>
#include <boost/numeric/linear_algebra/concept_maps.hpp>

namespace mtl {

// Dispatching between simple and unrolled version
template <std::ForwardIterator Iter, std::CopyConstructible Value, typename Op>
     requires std::Convertible<std::ForwardIterator<Iter>::value_type, Value>
           && std::Callable2<Op, Value, Value>
           && std::MoveAssignable<Value, std::Callable2<Op, Value, Value>::result_type>
Value inline accumulate(Iter first, Iter last, Value init, Op op)
{
    std::cout << "Simple accumulate\n";

    for (; first != last; ++first)
	init= op(init, Value(*first));
    return init;
}


template <std::RandomAccessIterator Iter, std::CopyConstructible Value, typename Op>
  requires std::Convertible<std::RandomAccessIterator<Iter>::value_type, Value>
        && std::Callable2<Op, Value, Value>
        && std::MoveAssignable<Value, std::Callable2<Op, Value, Value>::result_type>
        && math::Commutative<Op, Value> 
        && math::Monoid<Op, Value> 
        && std::Convertible<math::Monoid<Op, Value>::identity_result_type, Value>
Value inline accumulate(Iter first, Iter last, Value init, Op op)
{
    std::cout << "Unrolled accumulate\n";

    typedef typename std::RandomAccessIterator<Iter>::value_type value_type;
    typedef typename std::RandomAccessIterator<Iter>::difference_type difference_type;
    Value             t0= identity(op, init), t1= identity(op, init), 
	              t2= identity(op, init), t3= init;
    difference_type   size= last - first, bsize= size >> 2 << 2, i;
    
    for (i= 0; i < bsize; i+= 4) {
	t0= op(t0, Value(first[i]));
	t1= op(t1, Value(first[i+1]));
	t2= op(t2, Value(first[i+2]));
	t3= op(t3, Value(first[i+3]));
    }
    for (; i < size; i++)
	t0= op(t0, Value(first[i]));

    t0= op(t0, t1), t2= op(t2, t3), t0= op(t0, t2);
    return t0;
}

} // namespace mtl

#endif // MTL_ACCUMULATION_TEST_INCLUDE
