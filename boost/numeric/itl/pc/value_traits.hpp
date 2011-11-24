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
//
// Algorithm inspired by Nick Vannieuwenhoven, written by Cornelius Steinhardt


#ifndef MTL_VALUE_TRAITS_INCLUDE
#define MTL_VALUE_TRAITS_INCLUDE

namespace mtl {
namespace traits {

template< typename V >
struct value_traits {
	static V zero;
	static V one;
};

}}

template<typename V> V mtl::traits::value_traits<V>::zero = V(0);
template<typename V> V mtl::traits::value_traits<V>::one = V(1);


#endif // MTL_VALUE_TRAITS_INCLUDE
