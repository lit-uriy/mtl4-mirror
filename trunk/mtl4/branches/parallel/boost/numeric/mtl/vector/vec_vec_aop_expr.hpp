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


// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_AOP_EXPR_INCLUDE
#define MTL_VEC_VEC_AOP_EXPR_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/numeric/mtl/vector/vec_expr.hpp>
#include <boost/numeric/mtl/operation/static_size.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>
#include <boost/numeric/mtl/operation/local.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/is_static.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/unroll_size1.hpp>
#include <boost/numeric/mtl/utility/with_unroll1.hpp>

namespace mtl { namespace vector {

    namespace impl {

	template <unsigned long Index, unsigned long Max, typename SFunctor>
	struct assign
	{
	    typedef assign<Index+1, Max, SFunctor>     next;

	    template <typename E1, typename E2, typename Size>
	    static inline void apply(E1& first, const E2& second, Size i)
	    {
		SFunctor::apply( first(i+Index), second(i+Index) );
		next::apply( first, second, i );
	    }
	};

	template <unsigned long Max, typename SFunctor>
	struct assign<Max, Max, SFunctor>
	{
	    template <typename E1, typename E2, typename Size>
	    static inline void apply(E1& first, const E2& second, Size i)
	    {
		SFunctor::apply( first(i+Max), second(i+Max) );
	    }
	};
    }

// Generic assign operation expression template for vectors
// Model of VectorExpression
template <typename E1, typename E2, typename SFunctor>
struct vec_vec_aop_expr 
  : public vec_expr< vec_vec_aop_expr<E1, E2, SFunctor> >
{
    typedef vec_expr< vec_vec_aop_expr<E1, E2, SFunctor> >  expr_base;
    typedef vec_vec_aop_expr<E1, E2, SFunctor>   self;
    typedef typename E1::value_type              value_type;
    
    typedef typename E1::size_type               size_type;
    typedef value_type reference_type ;

    typedef E1 first_argument_type ;
    typedef E2 second_argument_type ;
    
    vec_vec_aop_expr( first_argument_type& v1, second_argument_type const& v2 )
      : first( v1 ), second( v2 ), delayed_assign( false )
    {
	second.delay_assign();
    }
    
  private:
    void dynamic_assign(boost::mpl::false_) // Without unrolling
    {
	for (size_type i= 0; i < size(first); ++i)
	    SFunctor::apply( first(i), second(i) );
    }

    void dynamic_assign(boost::mpl::true_) // With unrolling
    {
	const size_type BSize= traits::unroll_size1<E1>::value0;
	size_type s= size(first), sb= s / BSize * BSize;

	for (size_type i= 0; i < sb; i+= BSize)
	    impl::assign<0, BSize-1, SFunctor>::apply(first, second, i);

	for (size_type i= sb; i < s; i++) 
	    SFunctor::apply( first(i), second(i) );
    }    


    void assign(boost::mpl::false_)
    {
	// If target is constructed by default it takes size of source
	if (size(first) == 0) first.change_dim(size(second));
	MTL_DEBUG_THROW_IF(size(first) != size(second), incompatible_size()); // otherwise error

	// need to do more benchmarking before making unrolling default
	dynamic_assign(traits::with_unroll1<E1>());
    }

    void assign(boost::mpl::true_)
    {
	MTL_DEBUG_THROW_IF(size(first) != size(second), incompatible_size()); // We cannot resize, only check
	
	// impl::assign<0, static_size<E1>::value-1, SFunctor>::apply(first, second); // Slower, at least on gcc
	for (size_type i= 0; i < size(first); ++i) // Do an ordinary loop instead
	    SFunctor::apply( first(i), second(i) );
    }

    // Non-distributed version
    void destroy(tag::universe)
    {
	// If target is constructed by default it takes size of source
	if (size(first) == 0) first.change_dim(size(second));

	// If sizes are different for any other reason, it's an error
	// std::cout << "~vec_vec_aop_expr() " << size(first) << "  " << size(second) << std::endl;
	MTL_DEBUG_THROW_IF(size(first) != size(second), incompatible_size());
	
	for (size_type i= 0; i < size(first); ++i)
	    SFunctor::apply( first(i), second(i) );

	// Slower, at least on gcc
	// assign(traits::is_static<E1>());
    }

    // Distributed version
    void destroy(tag::distributed)
    {
	// Check distribution !!!

	typedef typename DistributedCollection<E1>::local_type LocalE1;
	typedef typename DistributedCollection<E2>::local_type LocalE2;
	// Create and destroy local expression so that local operation is performed here
	// std::cout << "local(first) is " << local(first) << ", local(second).size() is " << local(second).size() << std::endl;

	LocalE2 local_second(local(second));
	vec_vec_aop_expr<LocalE1, LocalE2, SFunctor> local_tmp(local(first), local_second);
	// int dummy= 4; // for debugging
    }

  public:
    ~vec_vec_aop_expr() { if (!delayed_assign) destroy(typename mtl::traits::category<E1>::type()); }
    
    void delay_assign() const { delayed_assign= true; }

    friend size_type inline size(const self& x)
    {
	assert( size(x.first) == 0 || size(x.first) == size(x.second) );
	return size(x.second);
    }

    value_type& operator() ( size_type i ) const
    {
	assert( delayed_assign );
	return SFunctor::apply( first(i), second(i) );
    }

    value_type& operator[] ( size_type i ) const { return (*this)(i); }
    
#if 0
    // Might need refactoring
    template <typename EE1, typename EE2, typename SSFunctor> 
    friend typename DistributedVector< vec_vec_aop_expr<EE1, EE2, SSFunctor> >::distribution_type
    distribution(const vec_vec_aop_expr<EE1, EE2, SSFunctor>& expr);
    {
	MTL_DEBUG_THROW_IF(distribution(expr.first) != distribution(expr.second), incompatible_distribution());
	return distribution(expr.first);
    }
#endif

  public:
     mutable first_argument_type&        first ;
     second_argument_type const&         second ;
  private:
     mutable bool                        delayed_assign;
}; // vec_vec_aop_expr

#if 0
template <typename E1, typename E2, typename Functor>
typename mtl::traits::distribution<vec_vec_aop_expr<E1, E2, Functor> >::type
inline distribution(const vec_vec_aop_expr<E1, E2, Functor>& v)
{
    MTL_DEBUG_THROW_IF(distribution(v.first) != distribution(v.second), incompatible_distribution());
    return distribution(v.reference_first());
}
#endif

} } // Namespace mtl::vector





#endif


