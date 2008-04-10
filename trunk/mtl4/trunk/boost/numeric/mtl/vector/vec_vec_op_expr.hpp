// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_OP_EXPR_INCLUDE
#define MTL_VEC_VEC_OP_EXPR_INCLUDE

#include <boost/numeric/mtl/vector/vec_expr.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <typename E1, typename E2, typename SFunctor>
struct vec_vec_op_expr 
   : public vec_expr< vec_vec_op_expr<E1, E2, SFunctor> >
{
    typedef vec_expr< vec_vec_op_expr<E1, E2, SFunctor> > expr_base;
    typedef vec_vec_op_expr                     self;

    // temporary solution
    // typedef typename E1::value_type              value_type;
    // typedef typename glas::value< glas::scalar::vec_vec_op_expr<typename E1::value_type, typename E2::value_type > >::type value_type ;
    typedef typename SFunctor::result_type       value_type;

    // temporary solution
    typedef typename E1::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type                           const_dereference_type ;

    typedef E1                                   first_argument_type ;
    typedef E2                                   second_argument_type ;
    
public:
    vec_vec_op_expr( first_argument_type const& v1, second_argument_type const& v2 )
	: first( v1 ), second( v2 )
    {
	std::cerr << "vec_vec_op_expr.vec_vec_op_expr()  " << first.size() << "  " << second.size() << "\n";

	first.delay_assign();
	second.delay_assign();
    }
    
    void delay_assign() const {}

    size_type size() const
    {
	std::cerr << "vec_vec_op_expr.size() " << first.size() << "  " << second.size() << "\n";
	assert( first.size() == second.size() ) ;
	return first.size() ;
    }

    const_dereference_type operator() ( size_type i ) const
    {
        return SFunctor::apply( first( i ), second( i ) ) ;
        // return first( i ) + second( i ) ;
    }

    const_dereference_type operator[] ( size_type i ) const
    {
        return SFunctor::apply( first( i ), second( i ) ) ;
        // return first( i ) + second( i ) ;
    }

  private:
    first_argument_type const&  first ;
    second_argument_type const& second ;
} ; // vec_vec_op_expr

    
// Model of VectorExpression
template <typename E1, typename E2, typename SFunctor>
struct vecs_vecs_op_expr 
   : public vec_expr< vecs_vecs_op_expr<E1, E2, SFunctor> >
{
    typedef vec_expr< vecs_vecs_op_expr<E1, E2, SFunctor> > expr_base;
    typedef vecs_vecs_op_expr                     self;

    // temporary solution
    // typedef typename E1::value_type              value_type;
    // typedef typename glas::value< glas::scalar::vec_vec_op_expr<typename E1::value_type, typename E2::value_type > >::type value_type ;
    typedef typename SFunctor::result_type       value_type;

    // temporary solution
    typedef typename E1::type::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type                           const_dereference_type ;

    typedef E1                                   first_argument_type ;
    typedef E2                                   second_argument_type ;
    
public:
    vecs_vecs_op_expr( first_argument_type const& v1, second_argument_type const& v2 )
	: first( v1 ), second( v2 )
    {
	// std::cerr << "vec_vec_op_expr.vec_vec_op_expr()  " << first.size() << "  " << second.size() << "\n";
	first.value.delay_assign();
	second.value.delay_assign();
    }
    
    void delay_assign() const {}

    size_type size() const
    {
	// std::cerr << "vec_vec_op_expr.size() " << first.value.size() << "  " << second.value.size() << "\n";
	assert( first.value.size() == second.value.size() ) ;
	return first.value.size() ;
    }

    const_dereference_type operator() ( size_type i ) const
    {
        return SFunctor::apply( first.value( i ), second.value( i ) ) ;
        // return first( i ) + second( i ) ;
    }

    const_dereference_type operator[] ( size_type i ) const
    {
        return SFunctor::apply( first.value( i ), second.value( i ) ) ;
        // return first( i ) + second( i ) ;
    }

  private:
    first_argument_type const&  first ;
    second_argument_type const& second ;
} ; // vec_vec_op_expr

    
} } // Namespace mtl::vector


#endif

