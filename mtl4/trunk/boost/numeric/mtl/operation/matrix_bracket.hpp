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

#ifndef MTL_MATRIX_BRACKETS_INCLUDE
#define MTL_MATRIX_BRACKETS_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl { namespace operations {

    template <typename Matrix, typename Ref, typename ValueRef>
    struct bracket_proxy
    {
	typedef typename Matrix::value_type   value_type;
	typedef typename Matrix::size_type    size_type;
	// typedef typename Matrix::row_vector_type vector_type;
	typedef dense_vector<value_type, mtl::vector::parameters<mtl::col_major> > vector_type;

	explicit bracket_proxy(Ref matrix, size_type row) : matrix(matrix), row(row) {}

	ValueRef operator[] (size_type col)
	{
	    return matrix(row, col);
	}

#if 0
	typename Matrix::row_vector_type operator[] (const irange& col_range)
	{
	    return matrix.sub_vector(row, col_range);
	}
#endif

#if 0
	const ValueRef operator[] (size_type col) const
	{
	    return matrix(row, col);
	}
#endif

      protected:
	Ref         matrix;
	size_type   row;
    };


    template <typename Matrix, typename Ref, typename ValueRef>
    struct range_bracket_proxy
    {
	explicit range_bracket_proxy(Ref matrix, irange row_range) : matrix(matrix), row_range(row_range) {}

	ValueRef operator[] (const irange& col_range)
	{
	    return sub_matrix(matrix, row_range.start(), row_range.finish(),
			      col_range.start(), col_range.finish());
	}
#if 0
	typename Matrix::col_vector_type operator[] (size_type col)
	{
	    return matrix.sub_vector(row_range, col);
	}
#endif
      protected:
	Ref         matrix;
	irange      row_range;
    };





} // namespace operations

} // NAMESPACE mtl

#endif // MTL_MATRIX_BRACKETS_INCLUDE
