// $COPYRIGHT$

#ifndef MTL_FROBENIUS_NORM_INCLUDE
#define MTL_FROBENIUS_NORM_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/operation/max_of_sums.hpp>


namespace mtl {

/// Frobenius norm, i.e. square root of sum of squares of all entries sqrt(sum_i sum_j(|a[i][j]|^2))
template <typename Matrix>
typename RealMagnitude<typename Collection<Matrix>::value_type>::type
inline frobenius_norm(const Matrix& matrix)
{
    using std::sqrt; using std::abs; using math::zero;

    typename traits::const_value<Matrix>::type     value(matrix); 

    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename RealMagnitude<value_type>::type  real_type;
    real_type ref, sum= zero(ref);

    typedef typename traits::range_generator<tag::major, Matrix>::type     cursor_type;
    typedef typename traits::range_generator<tag::nz, cursor_type>::type   icursor_type;

    for (cursor_type cursor = begin<tag::major>(matrix), cend = end<tag::major>(matrix); cursor != cend; ++cursor) 
	for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor) {
	    real_type tmp= abs(value(*icursor));
	    sum+= tmp * tmp;
	}
    return sqrt(sum);
}

} // namespace mtl

#endif // MTL_FROBENIUS_NORM_INCLUDE
