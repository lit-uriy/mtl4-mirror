// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MEAN_INCLUDE
#define MTL_MEAN_INCLUDE


#include <boost/numeric/mtl/utility/enable_if.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl {
    
    namespace matrix {
	/// Return mean values from all colums in matrix in a vector of num_cols(A)
	template <typename Matrix>
	dense_vector<typename Collection<Matrix>::value_type>
	inline mean(const Matrix& A)
	{
	    namespace traits = mtl::traits;
	    using std::abs;
	    typedef typename Collection<Matrix>::value_type   value_type;
	    typedef typename Collection<Matrix>::size_type    size_type;
    
	    dense_vector<value_type> mean(num_cols(A));  mean= 0;
	    typename traits::row<Matrix>::type             row(A); 
	    typename traits::col<Matrix>::type             col(A); 
	    typename traits::const_value<Matrix>::type     value(A); 
	    typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	
	    for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor) {
		typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
		for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor) {
			mean[col(*icursor)]+= value(*icursor);
		}
	    }
	    if (num_rows(A) > 0)
		mean/= num_rows(A);
	
	    return mean;
	}
	
	/// Return mean values from all rows in matrix in a vector of num_rows(A)
	template <typename Matrix>
	dense_vector<typename Collection<Matrix>::value_type>
	inline mean(const Matrix& A, int i)
	{
	    namespace traits = mtl::traits;
	    using std::abs;
	    typedef typename Collection<Matrix>::value_type   value_type;
	    typedef typename Collection<Matrix>::size_type    size_type;
    
	    dense_vector<value_type> mean(num_rows(A));  mean= 0;
	    if (i!=2) return mean;
	    typename traits::row<Matrix>::type             row(A); 
	    typename traits::col<Matrix>::type             col(A); 
	    typename traits::const_value<Matrix>::type     value(A); 
	    typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	
	    for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor) {
		typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
		for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor) {
			mean[row(*icursor)]+= value(*icursor);
		}
	    }
	    if (num_cols(A) > 0)
		mean/= num_cols(A);
	
	    return mean;
	}

} // namespace matrix
    
    

    namespace vector {

	/// Return mean value from all vector elements
	template <typename Vector>
	typename mtl::traits::enable_if_vector<Vector, typename Collection<Vector>::value_type>::type
	inline mean(const Vector& v)
 	{
	    typedef typename Collection<Vector>::size_type    size_type;
	    typedef typename Collection<Vector>::value_type   value_type;

	    value_type mean(0);

	    for (size_type i= 0; i < size(v); i++)
		mean+= v[i];
	    if (size(v) > 0){
		mean/=size(v);
	    }
	    
	    return mean;
	}
    }
} // namespace mtl

#endif // MTL_MEAN_INCLUDE
