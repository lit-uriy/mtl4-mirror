// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_PLUS_EXPR_INCLUDE
#define MTL_VEC_VEC_PLUS_EXPR_INCLUDE

#include <boost/static_assert.hpp>

#include <boost/numeric/mtl/vector/vec_vec_op_expr.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>
#include <boost/numeric/mtl/operation/compute_summand.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
struct vec_vec_plus_expr 
    : vec_expr< vec_vec_plus_expr<E1, E2> >
{
    typedef typename operation::compute_summand<E1>::type    first_argument_type;
    typedef typename operation::compute_summand<E2>::type    second_argument_type;

    typedef typename sfunctor::plus<typename first_argument_type::value_type, 
				    typename second_argument_type::value_type>::result_type 
	const_dereference_type;
    typedef const_dereference_type                           value_type;

    typedef typename first_argument_type::size_type          size_type;


    typedef vec_vec_op_expr< first_argument_type, second_argument_type,
			     sfunctor::plus<typename E1::value_type, typename E2::value_type> > base;

    vec_vec_plus_expr( E1 const& v1, E2 const& v2 )
	: first(v1), second(v2)
    {}

    void delay_assign() const {}

    size_type size() const
    {
	// std::cerr << "vec_vec_plus_expr.size() " << first.value.size() << "  " << second.value.size() << "\n";
	assert( first.value.size() == second.value.size() ) ;
	return first.value.size() ;
    }

    const_dereference_type operator() (size_type i) const
    {
        return first.value(i) + second.value(i);
    }

    const_dereference_type operator[] (size_type i) const
    {
        return first.value[i] + second.value[i];
    }

  private:
    operation::compute_summand<E1> first;
    operation::compute_summand<E2> second;
};


template <typename E1, typename E2>
inline vec_vec_plus_expr<E1, E2>
operator+ (const vec_expr<E1>& e1, const vec_expr<E2>& e2)
{
    // do not add row and column vectors (or inconsistent value types)
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<E1>::type, 
			                typename ashape::ashape<E2>::type>::value));
    return vec_vec_plus_expr<E1, E2>(static_cast<const E1&>(e1), static_cast<const E2&>(e2));
}


} } // Namespace mtl::vector




#endif

