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

#ifndef MTL_TRAITS_REPLACE_VALUE_TYPE_INCLUDE
#define MTL_TRAITS_REPLACE_VALUE_TYPE_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/static_assert.hpp>
#include <boost/numeric/mtl/utility/is_what.hpp>

namespace mtl { namespace traits {

/// If first argument is a vector or matrix, its value_type is replaced by the second argument.
/** Otherwise it is a static error. **/
template <typename T, typename Value>
struct replace_value_type
{};

template <template <typename, typename> class Cont, typename OldValue, typename Parameter, typename NewValue>
struct replace_value_type<Cont<OldValue, Parameter>, NewValue>
{
    MTL_STATIC_ASSERT((is_matrix<Cont<OldValue, Parameter> >::value || is_vector<Cont<OldValue, Parameter> >::value),
		       "First template argument must be a matrix or vector.");
    typedef Cont<NewValue, Parameter> type;
};

template <typename OldValue, std::size_t Mask, typename Parameter, typename NewValue>
struct replace_value_type<mat::morton_dense<OldValue, Mask, Parameter>, NewValue>
{
    typedef mat::morton_dense<NewValue, Mask, Parameter> type;
};



}} // namespace mtl::traits

#endif // MTL_TRAITS_REPLACE_VALUE_TYPE_INCLUDE
