// $COPYRIGHT$

#ifndef MTL_MAT_MAT_TIMES_EXPR_INCLUDE
#define MTL_MAT_MAT_TIMES_EXPR_INCLUDE

#include <boost/numeric/mtl/matrix/mat_mat_op_expr.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>

namespace mtl { namespace matrix {

template <typename E1, typename E2>
struct mat_mat_times_expr 
    : public mat_mat_op_expr< E1, E2, sfunctor::times<typename E1::value_type, typename E2::value_type> >,
      public mat_expr< mat_mat_times_expr<E1, E2> >
{
    typedef mat_mat_op_expr< E1, E2, sfunctor::times<typename E1::value_type, typename E2::value_type> > op_base;
    typedef mat_expr< mat_mat_times_expr<E1, E2> >                                                       crtp_base;
    typedef E1                                   first_argument_type ;
    typedef E2                                   second_argument_type ;
    
    mat_mat_times_expr( E1 const& v1, E2 const& v2 )
	: op_base( v1, v2 ), crtp_base(*this), first(v1), second(v2)
    {}

    first_argument_type const&  first ;
    second_argument_type const& second ;
};


}} // Namespace mtl::matrix

#endif // MTL_MAT_MAT_TIMES_EXPR_INCLUDE
