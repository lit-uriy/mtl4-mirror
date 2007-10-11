// $COPYRIGHT$

#ifndef MTL_HERMITIAN_INCLUDE
#define MTL_HERMITIAN_INCLUDE

#include <boost/numeric/mtl/matrix/map_view.hpp>

namespace mtl { 

template <typename Matrix>
matrix::hermitian_view<Matrix> inline hermitian(const Matrix& matrix)
{
    return matrix::hermitian_view<Matrix>(matrix);
}


} // namespace mtl

#endif // MTL_HERMITIAN_INCLUDE
