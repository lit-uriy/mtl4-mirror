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
#include <boost/numeric/mtl/operation/column_in_matrix.hpp>
#include <boost/numeric/mtl/operation/row_in_matrix.hpp>

namespace mtl { namespace operations {

    template <typename T, bool can_irange, typename Ref, typename ValueRef> struct bracket_proxy_impl {};
    
    template <typename Ref, typename ValueRef>
    struct bracket_proxy_impl<irange, true, Ref, ValueRef>
    {
	typedef RowInMatrix<typename boost::remove_reference<Ref>::type> row_traits;
	typedef typename row_traits::type                                type;

	template <typename Size>
	type static inline apply(Ref matrix, Size row, irange col_range)
	{
	    return row_traits::apply(matrix, row, col_range);
	}
    };

    template <typename Matrix, typename Ref, typename ValueRef>
    struct bracket_proxy
    {
	typedef typename Matrix::value_type      value_type;
	typedef typename Matrix::size_type       size_type;
	typedef RowInMatrix<typename boost::remove_reference<Ref>::type> row_traits;

	explicit bracket_proxy(Ref matrix, size_type row) : matrix(matrix), row(row) {}

	ValueRef operator[] (size_type col) { return matrix(row, col);	}

#if 0
	template <typename T>
	typename bracket_proxy_impl<T, row_traits::exists, Ref, ValueRef>::type operator[] (const T& col_range)
	{ return bracket_proxy_impl<T, row_traits::exists, Ref, ValueRef>::apply(matrix, row, col_range); }
#endif

#if 1
	template <typename T> struct my_traits { static const bool value= boost::is_same<T, mtl::irange>::value && row_traits::exists; };

	template <typename T>
	typename boost::lazy_enable_if_c<my_traits<T>::value, row_traits>::type	 
	operator[] (const T& col_range) 
	{ 
	    return row_traits::apply(matrix, row, col_range); 
	}
#endif
      protected:
	Ref         matrix;
	size_type   row;
    };

#if 0 // Doesn't compile either -> to be deleted soon
    template <typename T, bool can_int, typename Ref, typename ValueRef> struct range_bracket_proxy_impl {};

#if 0
    template <typename Ref, typename ValueRef>
    struct range_bracket_proxy_impl<irange, false, Ref, ValueRef>
    {
	typedef ValueRef     type;
	type static inline apply(Ref matrix, const irange& row_range, const irange& col_range)
	{
	    return sub_matrix(matrix, row_range.start(), row_range.finish(),
			      col_range.start(), col_range.finish());
	}
    };
#endif

    template <typename T, typename Ref, typename ValueRef>
    struct range_bracket_proxy_impl<T, true, Ref, ValueRef>
    {
	typedef ColumnInMatrix<typename boost::remove_reference<Ref>::type> col_traits;
	typedef typename col_traits::type                                   type;

	type static inline apply(Ref matrix, const irange& row_range, const T& col)
	{
	    return col_traits::apply(matrix, row_range, col);
	}
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
	struct my_traits 
	{
	    static const bool value = boost::is_integral<T>::value && col_traits::exists;
	};

#if 1
	template <typename T> 
	typename range_bracket_proxy_impl<T, my_traits<T>::value, Ref, ValueRef>::type operator[] (const T& col) 
	{ return range_bracket_proxy_impl<T, my_traits<T>::value, Ref, ValueRef>::apply(matrix, row_range, col); }
#endif

#if 0
	template <typename T> 
	typename range_bracket_proxy_impl<T, boost::is_integral<T>::value && col_traits::exists, Ref, ValueRef>::type operator[] (const T& col) 
	{ return range_bracket_proxy_impl<T, boost::is_integral<T>::value && col_traits::exists, Ref, ValueRef>::apply(matrix, row_range, col); }
#endif

      protected:
	Ref         matrix;
	irange      row_range;
    };
#endif


#if 1 // Doesn't compile on Intel!!!
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
	struct my_traits 
	{
	    static const bool value = boost::is_integral<T>::value && col_traits::exists;
	};

	template <typename T>
	typename boost::lazy_enable_if_c<my_traits<T>::value, col_traits>::type	 
	operator[] (T col)  { return col_traits::apply(matrix, row_range, col); }

#if 0
	template <typename T>
	typename boost::lazy_enable_if_c<boost::is_integral<T>::value && col_traits::exists, col_traits>::type	 
	operator[] (T col)  { return col_traits::apply(matrix, row_range, col); }
#endif
      protected:
	Ref         matrix;
	irange      row_range;
    };
#endif




} // namespace operations

} // NAMESPACE mtl

#endif // MTL_MATRIX_BRACKETS_INCLUDE
