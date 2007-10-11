// $COPYRIGHT$

#ifndef MTL_ONE_NORM_INCLUDE
#define MTL_ONE_NORM_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/operation/max_of_sums.hpp>
#include <boost/numeric/mtl/vector/reduction.hpp>
#include <boost/numeric/mtl/vector/reduction_functors.hpp>


namespace mtl {

    namespace impl {

	// Ignore unrolling for matrices 
	template <unsigned long Unroll, typename Matrix>
	typename RealMagnitude<typename Collection<Matrix>::value_type>::type
	inline one_norm(const Matrix& matrix, tag::matrix)
	{
	    typename traits::col<Matrix>::type                             col(matrix); 
	    return max_of_sums(matrix, !traits::is_row_major<typename OrientedCollection<Matrix>::orientation>(), 
			       col, num_cols(matrix));
	}

	template <unsigned long Unroll, typename Vector>
	typename RealMagnitude<typename Collection<Vector>::value_type>::type
	inline one_norm(const Vector& vector, tag::vector)
	{
	    typedef typename RealMagnitude<typename Collection<Vector>::value_type>::type result_type;
	    return vector::reduction<Unroll, vector::one_norm_functor, result_type>::apply(vector);
	}
    }


template <unsigned long Unroll, typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline one_norm(const Value& value)
{
    return impl::one_norm<Unroll>(value, typename traits::category<Value>::type());
}

/*! One-norm for vectors and matrices: one_norm(x) \f$\rightarrow |x|_1\f$.
    \retval The magnitude type of the respective value type, see Magnitude.

    The norms are defined
    - For vectors: \f$|v|_1=\sum_i |v_i|\f$; and
    - For matrices: \f$|A|_1=\max_j\{\sum_i(|A_{ij}|)\}\f$.

    Vector norms are unrolled 8-fold by default. 
    An n-fold unrolling can be generated with one_norm<n>(x).
    The maximum for n is 8 (it might be increased later).
    Matrix norms are not (yet)
    optimized.
**/
template <typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline one_norm(const Value& value)
{
    return one_norm<8>(value);
}

} // namespace mtl

#endif // MTL_ONE_NORM_INCLUDE
