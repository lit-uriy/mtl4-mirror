// $COPYRIGHT$

#ifndef MTL_BIN_OP_EXPR_INCLUDE
#define MTL_BIN_OP_EXPR_INCLUDE

namespace mtl {

/// Minimalistic expression template for binary operation: keeps only references.
template <typename E1, typename E2>
struct bin_op_expr
{
    typedef bin_op_expr                          self;

    typedef E1                                   first_argument_type ;
    typedef E2                                   second_argument_type ;

    bin_op_expr( first_argument_type const& v1, second_argument_type const& v2 )
	: first( v1 ), second( v2 )
    {}

    first_argument_type const&  first ;
    second_argument_type const& second ;
};

} // namespace mtl

#endif // MTL_BIN_OP_EXPR_INCLUDE
