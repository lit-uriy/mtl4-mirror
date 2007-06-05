// $COPYRIGHT$

#ifndef MTL_TWO_NORM_INCLUDE
#define MTL_TWO_NORM_INCLUDE

#include <iostream>
#include <cmath>

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/vector/reduction.hpp>
#include <boost/numeric/mtl/vector/reduction_functors.hpp>


namespace mtl {

    namespace impl {

	// Ignore unrolling for matrices 
	template <unsigned long Unroll, typename Matrix>
	typename RealMagnitude<typename Collection<Matrix>::value_type>::type
	inline two_norm(const Matrix& matrix, tag::matrix)
	{
	    std::cout << "Volunteers to implement efficient two-norm of matrices still searched\n";
	    return 0.0;
	}
	
	template <unsigned long Unroll, typename Vector>
	typename RealMagnitude<typename Collection<Vector>::value_type>::type
	inline two_norm(const Vector& vector, tag::vector)
	{
	    using std::sqrt;
	    typedef typename RealMagnitude<typename Collection<Vector>::value_type>::type result_type;
	    return sqrt(vector::reduction<Unroll, vector::two_norm_functor, result_type>::apply(vector));
	}
	
    } // namespace impl


template <unsigned long Unroll, typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline two_norm(const Value& value)
{
    return impl::two_norm<Unroll>(value, typename traits::category<Value>::type());
}

template <typename Value>
typename RealMagnitude<typename Collection<Value>::value_type>::type
inline two_norm(const Value& value)
{
    return two_norm<8>(value);
}

} // namespace mtl

#endif // MTL_TWO_NORM_INCLUDE
