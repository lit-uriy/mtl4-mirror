// $COPYRIGHT$

#ifndef MTL_TRANS_INCLUDE
#define MTL_TRANS_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/matrix/transposed_view.hpp>

namespace mtl {

namespace sfunctor {

    // General case is not defined
    template <typename Value, typename AlgebraicCategory>
    struct trans {};

    template <typename Matrix>
    struct trans<Matrix, tag::matrix>

    {
	typedef transposed_view<Matrix> result_type;

	static inline result_type apply(const Matrix& matrix)
	{
	    // Needs improvement in the future !!!
	    return result_type(const_cast<Matrix&>(matrix));
	}

	result_type operator() (const Matrix& matrix) const
	{
	    return apply(matrix);
	}
    };

} // namespace sfunctor


template <typename Value>
typename sfunctor::trans<Value, typename traits::algebraic_category<Value>::type>::result_type 
inline trans(const Value& v)
{
    return sfunctor::trans<Value, typename traits::algebraic_category<Value>::type>::apply(v);
};


} // namespace mtl

#endif // MTL_TRANS_INCLUDE
