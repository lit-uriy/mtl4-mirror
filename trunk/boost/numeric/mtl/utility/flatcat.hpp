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

#ifndef MTL_TRAITS_FLATCAT_INCLUDE
#define MTL_TRAITS_FLATCAT_INCLUDE

#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>

namespace mtl { namespace traits {


template <typename C, typename U1>
struct flatcat1_c
  : boost::mpl::if_<boost::is_base_of<U1, C>,
		    tag::flat<U1>,
		    tag::universe
		   >::type
{};

template <typename T, typename U1>
struct flatcat1
  : flatcat1_c<typename category<T>::type, U1> {};

template <typename C, typename U1, typename U2>
struct flatcat2_c
  : boost::mpl::if_<boost::is_base_of<U1, C>,
		    tag::flat<U1>,
		    flatcat1_c<C, U2>
		   >::type
{};

template <typename T, typename U1, typename U2>
struct flatcat2
  : flatcat2_c<typename category<T>::type, U1, U2> {};

template <typename C, typename U1, typename U2, typename U3>
struct flatcat3_c
  : boost::mpl::if_<boost::is_base_of<U1, C>,
		    tag::flat<U1>,
		    flatcat2_c<C, U2, U3>
		   >::type
{};

template <typename T, typename U1, typename U2, typename U3>
struct flatcat3
  : flatcat3_c<typename category<T>::type, U1, U2, U3> {};


template <typename C, typename U1, typename U2, typename U3, typename U4>
struct flatcat4_c
  : boost::mpl::if_<boost::is_base_of<U1, C>,
		    tag::flat<U1>,
		    flatcat3_c<C, U2, U3, U4>
		   >::type
{};

template <typename T, typename U1, typename U2, typename U3, typename U4>
struct flatcat4
  : flatcat4_c<typename category<T>::type, U1, U2, U3, U4> {};




}} // namespace mtl::traits

#endif // MTL_TRAITS_FLATCAT_INCLUDE
