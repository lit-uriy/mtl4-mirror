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

#ifndef MTL_VEC_SCAL_AOP_EXPR_INCLUDE
#define MTL_VEC_SCAL_AOP_EXPR_INCLUDE

#include <boost/numeric/mtl/vector/vec_expr.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>
#include <boost/numeric/mtl/operation/local.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>


namespace mtl { namespace vector {

// Generic assign operation expression template for vectors
// Model of VectorExpression
template <class E1, class E2, typename SFunctor>
struct vec_scal_aop_expr 
    : public vec_expr< vec_scal_aop_expr<E1, E2, SFunctor> >
{
    typedef vec_expr< vec_scal_aop_expr<E1, E2, SFunctor> >  expr_base;
    typedef typename E1::value_type              value_type;
    
    // temporary solution
    typedef typename E1::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type reference_type ;

    typedef E1 first_argument_type ;
    typedef E2 second_argument_type ;
    
    vec_scal_aop_expr( first_argument_type& v1, second_argument_type const& v2 )
	  : first( v1 ), second( v2 ), delayed_assign( false )
    {}

  private:
    // Non-distributed version
    void destroy(tag::universe)
    {
	for (size_type i= 0; i < first.size(); ++i)
	    SFunctor::apply( first(i), second );
    }

    // Distributed version
    void destroy(tag::distributed)
    {
	typedef typename DistributedCollection<E1>::local_type LocalE1;
	// Create and destroy local expression so that local operation is performed here
	vec_scal_aop_expr<LocalE1, E2, SFunctor>(local(first), second);
    }

  public:
    ~vec_scal_aop_expr() { if (!delayed_assign) destroy(typename mtl::traits::category<E1>::type()); }
    
    void delay_assign() const { delayed_assign= true; }

    size_type size() const 
    {
	return first.size() ;
    }

    value_type& operator() ( size_type i ) const 
    {
	assert( delayed_assign );
	return SFunctor::apply( first(i), second );
    }

    value_type& operator[] ( size_type i ) const
    {
	assert( delayed_assign );
	return SFunctor::apply( first(i), second );
    }

  private:
     mutable first_argument_type&        first ;
     second_argument_type const&         second ;
     mutable bool                        delayed_assign;
  } ; // vec_scal_aop_expr

} } // Namespace mtl::vector





#endif

