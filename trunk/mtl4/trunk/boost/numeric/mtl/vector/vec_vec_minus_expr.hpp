// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_MINUS_EXPR_INCLUDE
#define MTL_VEC_VEC_MINUS_EXPR_INCLUDE

#include <boost/static_assert.hpp>

#include <boost/numeric/mtl/vector/vec_vec_op_expr.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/utility/sfunctor.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
struct vec_vec_minus_expr 
    : public vec_vec_op_expr< E1, E2, sfunctor::minus<typename E1::value_type, typename E2::value_type> >
{
    typedef vec_vec_op_expr< E1, E2, sfunctor::minus<typename E1::value_type, typename E2::value_type> > base;
    vec_vec_minus_expr( E1 const& v1, E2 const& v2 )
	: base( v1, v2 )
    {}
};

    
template <typename E1, typename E2>
inline vec_vec_minus_expr<E1, E2>
operator- (const vec_expr<E1>& e1, const vec_expr<E2>& e2)
{
    // do not minus row and column vectors (or inconsistent value types)
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<E1>::type, 
			                typename ashape::ashape<E1>::type>::value));
    return vec_vec_minus_expr<E1, E2>(e1.ref, e2.ref);
}


} } // Namespace mtl::vector




#endif

