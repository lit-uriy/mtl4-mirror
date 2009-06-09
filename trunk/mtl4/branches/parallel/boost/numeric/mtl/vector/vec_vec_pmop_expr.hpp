// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_VECTOR_VEC_VEC_PMOP_EXPR_INCLUDE
#define MTL_VECTOR_VEC_VEC_PMOP_EXPR_INCLUDE

#include <boost/numeric/mtl/vector/vec_expr.hpp>
#include <boost/numeric/mtl/operation/compute_summand.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2, typename SFunctor>
struct vec_vec_pmop_expr
  : vec_expr< vec_vec_pmop_expr<E1, E2, SFunctor> >
{
    typedef typename mtl::operation::compute_summand<E1>::type    first_argument_type;
    typedef typename mtl::operation::compute_summand<E2>::type    second_argument_type;
    typedef SFunctor                                              functor_type;

    typedef typename SFunctor::result_type                        const_dereference_type;

    typedef const_dereference_type                                value_type;
    typedef typename first_argument_type::size_type               size_type;

    vec_vec_pmop_expr( E1 const& v1, E2 const& v2 ) 
      : first(v1), second(v2)
    {
	first.value.delay_assign(); second.value.delay_assign();
    }

    void delay_assign() const {}

    size_type size() const
    {
	// std::cerr << "vec_vec_pmop_expr.size() " << first.value.size() << "  " << second.value.size() << "\n";
	assert( first.value.size() == second.value.size() ) ;
	return first.value.size() ;
    }

    const_dereference_type operator() (size_type i) const
    {
        return SFunctor::apply(first.value(i), second.value(i));
    }

    const_dereference_type operator[] (size_type i) const
    {
        return SFunctor::apply(first.value(i), second.value(i));
    }

    template <typename EE1, typename EE2, typename SSFunctor> 
    friend typename DistributedCollection< vec_vec_pmop_expr<EE1, EE2, SSFunctor> >::local_type
    local(const vec_vec_pmop_expr<EE1, EE2, SSFunctor>& expr);

    // Might need refactoring
    template <typename EE1, typename EE2, typename SSFunctor> 
    friend typename DistributedVector< vector::vec_vec_pmop_expr<EE1, EE2, SSFunctor> >::distribution_type
    distribution(const vec_vec_pmop_expr<EE1, EE2, SSFunctor>& expr);
#if 0
    {
	MTL_DEBUG_THROW_IF(distribution(expr.first.value) != distribution(expr.second.value), incompatible_distribution());
	return distribution(expr.first.value);
    }
#endif

  private:
    operation::compute_summand<E1> first;
    operation::compute_summand<E2> second;
};


}} // namespace mtl::vector

#endif // MTL_VECTOR_VEC_VEC_PMOP_EXPR_INCLUDE
