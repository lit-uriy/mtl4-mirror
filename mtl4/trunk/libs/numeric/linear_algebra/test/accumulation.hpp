#include <iostream>
#include <algorithm>
#include <concepts>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/new_concepts.hpp>

namespace mtl {

#if 0

template <std::ForwardIterator Iter, typename Value, typename Op>
     requires std::Convertible<Value, std::ForwardIterator<Iter>::value_type>
//&& math::Magma<Op, std::ForwardIterator<Iter>::value_type> )
typename std::ForwardIterator<Iter>::value_type 
inline accumulate_simple(Iter first, Iter last, Value init, Op op)
{
    typedef typename std::RandomAccessIterator<Iter>::value_type value_type;
    value_type        t0= init;
    
    // std::cout << "accumulate_simple\n";
    for (; first != last; ++first)
	t0= op(t0, *first);
    return t0;
}

template <typename Iter, typename Value, typename Op>
  requires std::RandomAccessIterator<Iter> 
	    && std::Convertible<Value, std::RandomAccessIterator<Iter>::value_type>
	    && math::CommutativeMonoid<Op, std::RandomAccessIterator<Iter>::value_type> )
typename std::RandomAccessIterator<Iter>::value_type 
inline accumulate_unrolled(Iter first, Iter last, Value init, Op op)
{
    typedef typename std::RandomAccessIterator<Iter>::value_type value_type;
    typedef typename std::RandomAccessIterator<Iter>::difference_type difference_type;
    value_type        t0= identity(op, init), t1= identity(op, init), t2= identity(op, init), t3= init;
    difference_type size= last - first, bsize= size >> 2 << 2, i;
    
    for (i= 0; i < bsize; i+= 4) {
	t0= op(t0, first[i]);
	t1= op(t1, first[i+1]);
	t2= op(t2, first[i+2]);
	t3= op(t3, first[i+3]);
    }
    for (; i < size; i++)
	t0= op(t0, first[i]);
    return op(op(t0, t1), op(t2, t3));
}

#endif

// Dispatching between simple and unrolled version
template <std::ForwardIterator Iter, std::Semiregular Value, typename Op>
     requires std::Convertible<Value, std::ForwardIterator<Iter>::value_type>
           && std::Callable2<Op, Value, std::ForwardIterator<Iter>::value_type>
           && std::MoveAssignable<Value,
				  std::Callable2<Op, Value, 
						 std::ForwardIterator<Iter>::value_type>::result_type>
//typename std::ForwardIterator<Iter>::value_type 
Value
inline accumulate(Iter first, Iter last, Value init, Op op)
{
    std::cout << "Simple accumulate\n";

    for (; first != last; ++first)
	init= op(init, *first);
    return init;
}

#if 0
template <typename Iter, typename Value, typename Op>
  requires std::RandomAccessIterator<Iter> 
	          && std::Convertible<Value, std::RandomAccessIterator<Iter>::value_type>
		  && math::CommutativeMonoid<Op, std::RandomAccessIterator<Iter>::value_type> )
typename std::RandomAccessIterator<Iter>::value_type 
inline accumulate(Iter first, Iter last, Value init, Op op)
{
    std::cout << "Unrolled accumulate\n";
    return accumulate_unrolled(first, last, init, op);
}

#endif


} // namespace mtl
