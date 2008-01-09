// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.


#ifndef MTL_VEC_SCAL_ASGN_EXPR_INCLUDE
#define MTL_VEC_SCAL_ASGN_EXPR_INCLUDE

#include <boost/static_assert.hpp>

#include <boost/numeric/mtl/vector/vec_scal_aop_expr.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
struct vec_scal_asgn_expr 
    : public vec_scal_aop_expr< E1, E2, sfunctor::assign<typename E1::value_type, E2> >
{
    typedef vec_scal_aop_expr< E1, E2, sfunctor::assign<typename E1::value_type, E2> > base;
    vec_scal_asgn_expr( E1& v1, E2 const& v2 )
	: base( v1, v2 )
    {}
};

} } // Namespace mtl::vector

#endif

