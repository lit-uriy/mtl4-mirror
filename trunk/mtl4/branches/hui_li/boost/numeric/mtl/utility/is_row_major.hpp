// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_IS_ROW_MAJOR_INCLUDE
#define MTL_IS_ROW_MAJOR_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>

namespace mtl { namespace traits {

    /// Meta-function whether a tag is row_major or col_major
    /** For convenience, directly applicable to matrix::parameter and vector::parameter. 
	Refined from boost::mpl::true_ or boost::mpl::false_ if defined.
    **/
    template <typename Parameter>
    struct is_row_major {};

    template <>
    struct is_row_major<row_major>
	: public boost::mpl::true_
    {};

    template <>
    struct is_row_major<col_major>
	: public boost::mpl::false_
    {};

    template <typename Index, typename Dimension, bool OnStack, bool RValue>
    struct is_row_major<matrix::parameters<row_major, Index, Dimension, OnStack, RValue> >
	: public boost::mpl::true_
    {};

    template <typename Index, typename Dimension, bool OnStack, bool RValue>
    struct is_row_major<matrix::parameters<col_major, Index, Dimension, OnStack, RValue> >
	: public boost::mpl::false_
    {};

    template <typename Dimension, bool OnStack, bool RValue>
    struct is_row_major<vector::parameters<row_major, Dimension, OnStack, RValue> >
	: public boost::mpl::true_
    {};

    template <typename Dimension, bool OnStack, bool RValue>
    struct is_row_major<vector::parameters<col_major, Dimension, OnStack, RValue> >
	: public boost::mpl::false_
    {};


}} // namespace mtl::traits

#endif // MTL_IS_ROW_MAJOR_INCLUDE
