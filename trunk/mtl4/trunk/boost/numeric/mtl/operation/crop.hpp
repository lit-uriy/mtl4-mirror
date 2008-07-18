// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_CROP_INCLUDE
#define MTL_CROP_INCLUDE

namespace mtl {

/// Remove all zero entries from a collection
/** Does nothing for dense collections **/
template <typename T>
void crop(T& x)
{
    x.crop();
}

} // namespace mtl

#endif // MTL_CROP_INCLUDE
