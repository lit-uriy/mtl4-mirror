// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG, www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also tools/license/license.mtl.txt in the distribution.

#ifndef MTL_TRAITS_AUTO_TYPE_H
#define MTL_TRAITS_AUTO_TYPE_H

#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/not.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/std_concept.hpp>
#include <boost/numeric/mtl/utility/common_matrix_type.hpp>
#include <boost/numeric/mtl/utility/replace_value_type.hpp>

namespace mtl { namespace traits { 

/// Automatic type evaluation, mostly from expression template to container
/** Containers evaluate to itself and expressions to a container that is 
    suited to hold the result of its evaluation. **/
template <typename T>
struct auto_type 
{
    typedef T type;
};

namespace detail {

    template <typename Matrix1, typename Matrix2>
    struct matrix_joiner
    {
	typedef typename auto_type<Matrix1>::type M1;
	typedef typename auto_type<Matrix2>::type M2;
	typedef typename common_matrix_type<M1, M2>::type type;
    };

}	

template <typename Matrix1, typename Matrix2>
struct auto_type<mtl::mat::mat_mat_plus_expr<Matrix1, Matrix2> >
  : detail::matrix_joiner<Matrix1, Matrix2>
{};

template <typename Matrix1, typename Matrix2>
struct auto_type<mtl::mat::mat_mat_minus_expr<Matrix1, Matrix2> >
  : detail::matrix_joiner<Matrix1, Matrix2>
{};

template <typename Matrix1, typename Matrix2>
struct auto_type<mtl::mat::mat_mat_times_expr<Matrix1, Matrix2> >
  : detail::matrix_joiner<Matrix1, Matrix2>
{};

template <typename Matrix1, typename Matrix2>
struct auto_type<mtl::mat::mat_mat_ele_times_expr<Matrix1, Matrix2> >
  : detail::matrix_joiner<Matrix1, Matrix2>
{};

template <typename Scaling, typename Vector>
struct auto_type<mtl::vec::scaled_view<Scaling, Vector> >
{
    typedef typename Multiplicable<Scaling, typename Collection<Vector>::value_type>::result_type value_type;
    typedef typename replace_value_type<typename auto_type<Vector>::type, value_type>::type type;
};

template <typename Matrix, typename Vector>
struct auto_type<mtl::mat_cvec_times_expr<Matrix, Vector> >
{
    typedef typename auto_type<Vector>::type AutoVector;
    typedef typename Multiplicable<typename Collection<Matrix>::value_type, 
				   typename Collection<Vector>::value_type>::result_type value_type;
    typedef typename replace_value_type<AutoVector, value_type>::type type;
};


#ifdef MTL_WITH_TEMPLATE_ALIAS
template <typename T>
using auto_t= typename auto_type<T>::type;
#endif

/// Meta-Predicate whether T has a auto type (different from itself) 
template <typename T>
struct has_auto_type 
  : boost::mpl::not_< boost::is_same<T, typename auto_type<T>::type> >
{};


}} // mtl::traits

#endif // MTL_TRAITS_AUTO_TYPE_H
