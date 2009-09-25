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

#include <algorithm>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/column_in_matrix.hpp>
#include <boost/numeric/mtl/concept/row_in_matrix.hpp>

namespace mtl { namespace operations {

    template <typename Matrix, typename Ref, typename ValueRef>
    struct bracket_proxy
    {
	typedef typename Matrix::value_type      value_type;
	typedef typename Matrix::size_type       size_type;
	typedef RowInMatrix<typename boost::remove_reference<Ref>::type> row_traits;

	explicit bracket_proxy(Ref matrix, size_type row) : matrix(matrix), row(row) {}

	ValueRef operator[] (size_type col) { return matrix(row, col);	}

	template <typename T>
	typename boost::lazy_enable_if_c<boost::is_same<T, mtl::irange>::value && row_traits::exists, row_traits>::type	 
	operator[] (const T& col_range) 
	{ 
	    return row_traits::apply(matrix, row, col_range); 
	}

#if 0
      private:

	size_type vector_size(const irange& col_range)
	{
	    using std::min;
	    size_type finish= min(col_range.finish(), num_cols(matrix));
	    return col_range.start() < finish ? finish - col_range.start() : 0;
	}

	vector_type dispatch(const irange& col_range, boost::mpl::true_)
	{
	    return vector_type(vector_size(col_range), &matrix[row][col_range.start()]);
	}

	vector_type dispatch(const irange& col_range, boost::mpl::false_)
	{
	    return vector_type(vector_size(col_range), &matrix[row][col_range.start()], );
	}	

      public:
	vector_type operator[] (const irange& col_range)
	{
	    return matrix.sub_vector(row, col_range);
	}

	//boost::mpl::bool_<RowInMatrix<Ref>::contiguous>


	typename Matrix::row_vector_type operator[] (const irange& col_range)
	{
	    return matrix.sub_vector(row, col_range);
	}


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
	typedef typename Matrix::size_type       size_type;
	typedef ColumnInMatrix<typename boost::remove_reference<Ref>::type> col_traits;

	explicit range_bracket_proxy(Ref matrix, const irange& row_range) : matrix(matrix), row_range(row_range) {}

	ValueRef operator[] (const irange& col_range)
	{
	    return sub_matrix(matrix, row_range.start(), row_range.finish(),
			      col_range.start(), col_range.finish());
	}

	template <typename T>
	typename boost::lazy_enable_if_c<boost::is_same<T, size_type>::value && col_traits::exists, col_traits>::type	 
	operator[] (size_type col)  { return col_traits::apply(matrix, row_range, col); }

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
