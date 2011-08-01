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

#ifndef MTL_EVALUATE_LAZY_INCLUDE
#define MTL_EVALUATE_LAZY_INCLUDE

#include <boost/numeric/mtl/operation/lazy_assign.hpp>


namespace mtl {

/// Overloaded function to evaluate lazy expressions
template <typename T, typename U, typename Assign>
void inline evaluate_lazy(lazy_assign<T, U, Assign>& lazy)
{
    Assign::first_update(lazy.first, lazy.second);
}

} // namespace mtl

#endif // MTL_EVALUATE_LAZY_INCLUDE
