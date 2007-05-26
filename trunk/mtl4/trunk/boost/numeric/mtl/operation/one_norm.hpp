// $COPYRIGHT$

#ifndef MTL_ONE_NORM_INCLUDE
#define MTL_ONE_NORM_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/operation/max_of_sums.hpp>


namespace mtl {

template <typename Matrix>
typename RealMagnitude<typename Collection<Matrix>::value_type>::type
inline one_norm(const Matrix& matrix)
{
    typename traits::col<Matrix>::type                             col(matrix); 
    return impl::max_of_sums(matrix, !traits::is_row_major<typename OrientedCollection<Matrix>::orientation>(), 
			     col, num_cols(matrix));
}

} // namespace mtl

#endif // MTL_ONE_NORM_INCLUDE
