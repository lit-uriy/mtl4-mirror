// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef META_MATH_ABS_INCLUDE
#define META_MATH_ABS_INCLUDE

namespace meta_math {

template <long int x>
struct abs
{
  static long int const value = x < 0 ? -x : x;
};


} // namespace meta_math

#endif // META_MATH_ABS_INCLUDE
