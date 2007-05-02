// $COPYRIGHT$

#ifndef MTL_MAT_EXPR_INCLUDE
#define MTL_MAT_EXPR_INCLUDE

namespace mtl { namespace matrix {

/// Base class for CRTP with matrices
template <typename Matrix>
struct mat_expr
{
    typedef Matrix   ref_type;

    explicit mat_expr(Matrix& ref) : ref(ref) {}

    ref_type&        ref;
};


/// Base class for CRTP with dense matrices
template <typename Matrix>
struct dmat_expr
    : public mat_expr<Matrix>
{
    typedef mat_expr<Matrix> base;

    explicit dmat_expr(Matrix& ref) : base(ref) {}
};


/// Base class for CRTP with sparse matrices
template <typename Matrix>
struct smat_expr
    : public mat_expr<Matrix>
{
    typedef mat_expr<Matrix> base;

    explicit smat_expr(Matrix& ref) : base(ref) {}
};


}} // namespace mtl::matrix

#endif // MTL_MAT_EXPR_INCLUDE
