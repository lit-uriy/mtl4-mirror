// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_SET_TO_0_INCLUDE
#define MTL_SET_TO_0_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl {

    namespace impl {

	template <class Matrix>
	void set_to_zero(Matrix& matrix, tag::contiguous_dense)
	{
	    using math::zero;
	    typename Matrix::value_type  ref, my_zero(zero(ref));
	    // std::cout << "set_to_zero: used_memory = " << matrix.used_memory() << "\n";
	    std::fill(matrix.elements(), matrix.elements()+matrix.used_memory(), my_zero);
#if 0
	    for (int i= 0; i < matrix.num_rows(); i++)
	      for (int j= 0; i < matrix.num_cols(); i++)
		matrix[i][j]= my_zero;
#endif
	}

	template <class Matrix>
	void set_to_zero(Matrix& matrix, tag::morton_dense)
	{
	    using math::zero;
	    typename Matrix::value_type  ref, my_zero(zero(ref));
	    // maybe faster to do it straight
	    // if performance problems we'll take care of the holes
	    // std::cout << "set_to_zero: used_memory = " << matrix.used_memory() << "\n";
	    std::fill(matrix.elements(), matrix.elements() + matrix.used_memory(), my_zero);

#if 0
	    for (int i= 0; i < matrix.num_rows(); i++)
	      for (int j= 0; i < matrix.num_cols(); i++)
		matrix[i][j]= my_zero;
#endif
	}	

	// Is approbriate for all sparse matrices and vectors
	template <class Matrix>
	void set_to_zero(Matrix& matrix, tag::sparse)
	{
	    matrix.make_empty();
	}
	
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
