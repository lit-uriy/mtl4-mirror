// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_ASGN_EXPR_INCLUDE
#define MTL_VEC_VEC_ASGN_EXPR_INCLUDE

#include <boost/static_assert.hpp>

#include <boost/numeric/mtl/vector/vec_vec_aop_expr.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
struct vec_vec_asgn_expr 
    : public vec_vec_aop_expr< E1, E2, sfunctor::assign<typename E1::value_type, typename E2::value_type> >
{
    typedef vec_vec_aop_expr< E1, E2, sfunctor::assign<typename E1::value_type, typename E2::value_type> > base;
    vec_vec_asgn_expr( E1& v1, E2 const& v2 )
	: base( v1, v2 )
    {}
};

} } // Namespace mtl::vector




#endif

