// $COPYRIGHT$

#ifndef MTL_MAT_CVEC_TIMES_EXPR_INCLUDE
#define MTL_MAT_CVEC_TIMES_EXPR_INCLUDE

#include <boost/numeric/mtl/operation/bin_op_expr.hpp>

namespace mtl {

template <typename E1, typename E2>
struct mat_cvec_times_expr 
    : public bin_op_expr< E1, E2 >
{
    typedef bin_op_expr< E1, E2 >   base;
    mat_cvec_times_expr( E1 const& matrix, E2 const& vector )
	: base(matrix, vector)
    {}
};

} // namespace mtl

#endif // MTL_MAT_CVEC_TIMES_EXPR_INCLUDE
