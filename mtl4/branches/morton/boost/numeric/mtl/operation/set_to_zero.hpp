// $COPYRIGHT$

#ifndef MTL_SET_TO_0_INCLUDE
#define MTL_SET_TO_0_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/traits.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl {

    namespace impl {

	template <class Matrix>
	void set_to_zero(Matrix& matrix, tag::contiguous_dense)
	{
	    using math::zero;
	    typename Matrix::value_type  ref, my_zero(zero(ref));
	    std::fill(matrix.elements(), matrix.elements()+matrix.used_memory(), my_zero);
	}

	template <class Matrix>
	void set_to_zero(Matrix& matrix, tag::morton_dense)
	{
	    using math::zero;
	    typename Matrix::value_type  ref, my_zero(zero(ref));
	    // maybe faster to do it straight
	    // if performance problems we'll take care of the holes
	    std::fill(matrix.elements(), matrix.elements() + matrix.used_memory(), my_zero);
	}	

	// TBD: sparse matrices by resetting the sparsity structure
	
    }

// Sets all values of a matrix to 0
// More spefically the defined multiplicative identity element
template <class Matrix>
void set_to_zero(Matrix& matrix)
{
    impl::set_to_zero(matrix, typename traits::category<Matrix>::type());
}
    

} // namespace mtl

#endif // MTL_SET_TO_0_INCLUDE
