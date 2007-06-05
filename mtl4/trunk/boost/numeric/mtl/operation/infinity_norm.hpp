// $COPYRIGHT$

#ifndef MTL_INFINITY_NORM_INCLUDE
#define MTL_INFINITY_NORM_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/operation/max_of_sums.hpp>
#include <boost/numeric/mtl/vector/reduction.hpp>
#include <boost/numeric/mtl/vector/reduction_functors.hpp>


namespace mtl {

    namespace impl {

	// Ignore unrolling for matrices 
	template <unsigned long Unroll, typename Matrix>
	typename RealMagnitude<typename Collection<Matrix>::value_type>::type
	inline infinity_norm(const Matrix& matrix, tag::matrix)
	{
	    typename traits::row<Matrix>::type                             row(matrix); 
	    return impl::max_of_sums(matrix, traits::is_row_major<typename OrientedCollection<Matrix>::orientation>(), 
				     row, num_rows(matrix));
	}

	template <unsigned long Unroll, typename Vector>
	typename RealMagnitude<typename Collection<Vector>::value_type>::type
	inline infinity_norm(const Vector& vector, tag::vector)
	{
	    typedef typename RealMagnitude<typename Collection<Vector>::value_type>::type result_type;
	    return vector::reduction<Unroll, vector::infinity_norm_functor, result_type>::apply(vector);
	}
	
    } // namespace impl


template <unsigned long Unroll, typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline infinity_norm(const Value& value)
{
    return impl::infinity_norm<Unroll>(value, typename traits::category<Value>::type());
}	

template <typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline infinity_norm(const Value& value)
{
    return infinity_norm<8>(value);
}

} // namespace mtl

#endif // MTL_INFINITY_NORM_INCLUDE
