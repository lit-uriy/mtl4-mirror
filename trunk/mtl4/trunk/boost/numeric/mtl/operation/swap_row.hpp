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

#ifndef MTL_MATRIX_SWAP_ROW
#define MTL_MATRIX_SWAP_ROW

#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/enable_if.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>


namespace mtl { 

namespace matrix {

    namespace detail {

	template <typename Matrix, typename Orientation>
	typename Collection<Matrix>::value_type
	inline swap_row(Matrix& A, typename Collection<Matrix>::size_type i,
			typename Collection<Matrix>::size_type j, tag::dense, Orientation)
	{
	    // swap(A[irange(i,i+1)][iall], A[irange(j,j+1)][iall];
	    using std::swap;
	    for (typename Collection<Matrix>::size_type k= 0; k < num_cols(A); k++)
		swap(A[i][k], A[j][k]);
	}
	
	template <typename Matrix>
	typename Collection<Matrix>::value_type
	inline swap_row(Matrix& A, typename Collection<Matrix>::size_type i,
			typename Collection<Matrix>::size_type j, tag::sparse, boost::mpl::true_)
	{
	    MTL_THROW(logic_error("This is not implemented yet."));
	}

	template <typename Matrix>
	typename Collection<Matrix>::value_type
	inline swap_row(Matrix& A, typename Collection<Matrix>::size_type i,
			typename Collection<Matrix>::size_type j, tag::sparse, boost::mpl::false_)
	{
	    MTL_THROW(logic_error("This is an ugly operation and not implemented yet."));
	}

    }

    template <typename Matrix>
    typename mtl::traits::enable_if_matrix<Matrix>::type
    inline swap_row(Matrix& A, typename Collection<Matrix>::size_type i,
		    typename Collection<Matrix>::size_type j)
    {
	detail::swap_row(A, i, j, typename mtl::traits::category<Matrix>::type(), 
			 mtl::traits::is_row_major<Matrix>());
    }

} // namespace matrix


namespace vector {

    template <typename Vector>
    typename mtl::traits::enable_if_vector<Vector>::type
    inline swap_row(Vector& v, typename Collection<Vector>::size_type i,
		    typename Collection<Vector>::size_type j)
    {
	using std::swap;
	swap(v[i], v[j]);
    }

} // vector

using matrix::swap_row;
using vector::swap_row;

} // namespace mtl

#endif // MTL_MATRIX_SWAP_ROW
