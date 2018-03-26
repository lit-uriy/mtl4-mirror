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

#ifndef MTL_TRAITS_COMMON_MATRIX_TYPE_INCLUDE
#define MTL_TRAITS_COMMON_MATRIX_TYPE_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/std_concept.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/parameters.hpp>
#include <boost/numeric/mtl/utility/is_what.hpp>
#include <boost/numeric/mtl/utility/static_assert.hpp>
#include <boost/numeric/mtl/utility/replace_value_type.hpp>

namespace mtl { namespace traits {

template <typename Matrix1, typename Matrix2>
struct same_matrix_template
  : boost::mpl::false_
{};

template <template <typename, typename> class Matrix, typename Value1, typename Parameter1, typename Value2, typename Parameter2>
struct same_matrix_template<Matrix<Value1, Parameter1>, Matrix<Value2, Parameter2> >
  : is_matrix<Matrix<Value1, Parameter1> >
{};

template <typename Value1, std::size_t Mask1, typename Parameter1, typename Value2, std::size_t Mask2, typename Parameter2>
struct same_matrix_template<mat::morton_dense<Value1, Mask1, Parameter1>, mat::morton_dense<Value2, Mask2, Parameter2> >
  : boost::mpl::true_
{};

template <typename Matrix1, typename Matrix2>
struct common_matrix_type
{
    MTL_STATIC_ASSERT((is_matrix<Matrix1>::value), "First template argument must be a matrix.");
    MTL_STATIC_ASSERT((is_matrix<Matrix2>::value), "Second template argument must be a matrix.");
    typedef typename Addable<typename Collection<Matrix1>::value_type, typename Collection<Matrix2>::value_type>::result_type value_type;
    typedef typename parameters<Matrix1>::type Para1;
    typedef typename boost::mpl::if_<
	same_matrix_template<Matrix1, Matrix2>,
	typename replace_value_type<Matrix1, value_type>::type,
	typename boost::mpl::if_c<
	    is_dense<Matrix1>::value || is_dense<Matrix2>::value,
	    mat::dense2D<value_type, Para1>,
	    mat::compressed2D<value_type, Para1>
	>::type
    >::type type;
};

}} // namespace mtl::traits

#endif // MTL_TRAITS_COMMON_MATRIX_TYPE_INCLUDE
