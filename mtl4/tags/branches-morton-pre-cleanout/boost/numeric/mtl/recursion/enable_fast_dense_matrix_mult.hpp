// $COPYRIGHT$

#ifndef MTL_ENABLE_FAST_DENSE_MATRIX_MULT_INCLUDE
#define MTL_ENABLE_FAST_DENSE_MATRIX_MULT_INCLUDE

#include <boost/numeric/mtl/dense2D.hpp>

namespace mtl { namespace recursion {

template <typename MatrixA, typename MatrixB, typename MatrixC>
struct enable_fast_dense_matrix_mult
{
    static const bool value= false;
};

template <typename E1, typename P1, typename E2, typename P2, typename E3, typename P3>
struct enable_fast_dense_matrix_mult<mtl::dense2D<E1, P1>, mtl::dense2D<E2, P2>, mtl::dense2D<E3, P3> >
{
    static const bool value= true;
};


}} // namespace mtl::recursion

#endif // MTL_ENABLE_FAST_DENSE_MATRIX_MULT_INCLUDE
