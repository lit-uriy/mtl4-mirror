// $COPYRIGHT$


#ifndef MTL_VEC_SCAL_TIMES_ASGN_EXPR_INCLUDE
#define MTL_VEC_SCAL_TIMES_ASGN_EXPR_INCLUDE

#include <boost/static_assert.hpp>

#include <boost/numeric/mtl/vector/vec_scal_aop_expr.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
struct vec_scal_times_asgn_expr 
    : public vec_scal_aop_expr< E1, E2, sfunctor::times_assign<typename E1::value_type, E2> >
{
    typedef vec_scal_aop_expr< E1, E2, sfunctor::times_assign<typename E1::value_type, E2> > base;
    vec_scal_times_asgn_expr( E1& v1, E2 const& v2 )
	: base( v1, v2 )
    {}
};

} } // Namespace mtl::vector

#endif

