// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_TRAITS_INDEX_EVALUATABLE_INCLUDE
#define MTL_TRAITS_INDEX_EVALUATABLE_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp> 

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/is_vector_reduction.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>

namespace mtl { namespace traits {

/// Type trait to check whether \p T can be evaluated index-wise (usually in lazy evaluation)
template <typename T>
struct index_evaluatable : boost::mpl::false_ {};

//#ifndef _MSC_VER // disable on Visual Studio until we know why it doesn't work there
template <typename T, typename U, typename Assign>
struct index_evaluatable<lazy_assign<T, U, Assign> >
  : boost::mpl::or_<
      boost::mpl::and_<is_vector<T>, is_scalar<U> >,
      boost::mpl::and_<is_vector<T>, is_vector<U> >,
      boost::mpl::and_<is_scalar<T>, is_vector_reduction<U> >
    >
{};

template <typename V1, typename Matrix, typename V2, typename Assign>
struct index_evaluatable<lazy_assign<V1, mtl::mat_cvec_times_expr<Matrix, V2>, Assign> >
  : is_row_major<Matrix> {};
//#endif

/// Type trait to control whether evaluation should be unrolled
template <typename T>
struct unrolled_index_evaluatable : boost::mpl::false_ {};

template <typename T, typename U, typename Assign>
struct unrolled_index_evaluatable<lazy_assign<T, U, Assign> >
  : boost::mpl::or_<
      boost::mpl::and_<is_vector<T>, is_scalar<U> >,
      boost::mpl::and_<is_vector<T>, is_vector<U> >,
      boost::mpl::and_<is_scalar<T>, is_vector_reduction<U> >
    >
{};

#if 0 // Even mat-vec-mult is faster when unrolled (on small matrices, on large it doesn't matter)
template <typename V1, typename Matrix, typename V2, typename Assign>
struct unrolled_index_evaluatable<lazy_assign<V1, mtl::mat_cvec_times_expr<Matrix, V2>, Assign> >
  : boost::mpl::false_ {};
#endif

/// Typetrait for forward evaluation
/** All index_evaluatable types are implicitly forward-evaluatable **/
template <typename T>
struct forward_index_evaluatable 
  : index_evaluatable<T>
{};

/// Typetrait for backward evaluation
/** All index_evaluatable types are implicitly backward-evaluatable **/
template <typename T>
struct backward_index_evaluatable 
  : index_evaluatable<T>
{};




}} // namespace mtl::traits

#endif // MTL_TRAITS_INDEX_EVALUATABLE_INCLUDE
