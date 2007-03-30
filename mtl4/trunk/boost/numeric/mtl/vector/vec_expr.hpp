// $COPYRIGHT$

#ifndef MTL_VEC_EXPR_INCLUDE
#define MTL_VEC_EXPR_INCLUDE

namespace mtl { namespace vector {

template <typename Vector>
struct vec_expr
{
    typedef Vector   ref_type;

    explicit vec_expr(Vector& ref) : ref(ref) {}

    ref_type&        ref;
};


}} // namespace mtl::vector

#endif // MTL_VEC_EXPR_INCLUDE
