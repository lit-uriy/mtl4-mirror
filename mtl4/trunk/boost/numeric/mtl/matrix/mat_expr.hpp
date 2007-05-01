// $COPYRIGHT$

#ifndef MTL_MAT_EXPR_INCLUDE
#define MTL_MAT_EXPR_INCLUDE

namespace mtl { namespace matrix {

template <typename Matrix>
struct mat_expr
{
    typedef Matrix   ref_type;

    explicit mat_expr(Matrix& ref) : ref(ref) {}

    ref_type&        ref;
};


}} // namespace mtl::matrix

#endif // MTL_MAT_EXPR_INCLUDE
