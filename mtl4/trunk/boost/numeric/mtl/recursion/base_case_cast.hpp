// $COPYRIGHT$

#ifndef MTL_BASE_CASE_CAST_INCLUDE
#define MTL_BASE_CASE_CAST_INCLUDE

#include <boost/numeric/mtl/recursion/base_case_matrix.hpp>
#include <boost/numeric/mtl/recursion/simplify_base_case_matrix.hpp>

namespace mtl { namespace recursion {


template <typename BaseCaseTest, typename Matrix>
typename base_case_matrix<Matrix, BaseCaseTest>::type inline
base_case_cast(Matrix const& matrix)
{
    return simplify_base_case_matrix(matrix, BaseCaseTest());
}


}} // namespace mtl::recursion

#endif // MTL_BASE_CASE_CAST_INCLUDE
