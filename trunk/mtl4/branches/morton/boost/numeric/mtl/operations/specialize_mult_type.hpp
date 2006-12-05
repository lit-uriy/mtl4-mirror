// $COPYRIGHT$

#ifndef MTL_SPECIALIZE_MULT_TYPE_INCLUDE
#define MTL_SPECIALIZE_MULT_TYPE_INCLUDE

namespace mtl {

// Specialize the functor that performs the multiplication to hand-tuned code under 2 conditions
// 1. It runs on a certain platform, given by conditional compilation with macros
// 2. The matrix has the appropriate type and memory layout
// Otherwise use the default functor
template <typename MatrixA, typename MatrixB, typename MatrixC, typename BaseCaseTest, typename DefaultMult>
struct specialize_mult_type
{
    typedef DefaultMult type;
}

} // namespace mtl

#include <boost/numeric/mtl/operations/specialize_mult_type_opteron.hpp>

#endif // MTL_SPECIALIZE_MULT_TYPE_INCLUDE
