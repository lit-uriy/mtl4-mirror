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

#ifndef MTL_TRAITS_AUTO_OR_CONST_REF_TYPE_H
#define MTL_TRAITS_AUTO_OR_CONST_REF_TYPE_H

#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/not.hpp>

#include <boost/numeric/mtl/utility/auto_type.hpp>

namespace mtl { namespace traits { 

/// Type trait that yields T's auto type if defined, otherwise const T&
template <typename T>
struct auto_or_const_ref_type
  : boost::mpl::if_<has_auto_type<T>, typename auto_type<T>::type, const T&> 
{};

#ifdef MTL_WITH_TEMPLATE_ALIAS
template <typename T>
using auto_or_const_ref_t= typename auto_or_const_ref_type<T>::type;
#endif


}} // mtl::traits

#endif // MTL_TRAITS_AUTO_OR_CONST_REF_TYPE_H
