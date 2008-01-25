// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef META_MATH_LOOP1_INCLUDE
#define META_MATH_LOOP1_INCLUDE

// See loop3.hpp for example

namespace meta_math {

template <unsigned long Index0, unsigned long Max0>
struct loop1
{
    static unsigned long const index0= Index0 - 1, next_index0= Index0 + 1;
};


template <unsigned long Max0>
struct loop1<Max0, Max0>
{
    static unsigned long const index0= Max0 - 1;
};


} // namespace meta_math

#endif // META_MATH_LOOP1_INCLUDE
