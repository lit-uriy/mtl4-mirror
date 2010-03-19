// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MEET_DATA_INCLUDE
#define MTL_MEET_DATA_INCLUDE

#include <cassert>

namespace mtl {

/// Move all data to consistent location, preferrably the device memory
/** If data of both arguments is on host and data is not too large
    leave it on the host. In all other cases move it to the device.
    Operations to move data are supposed to be const.
    \return Whether all data reside on the host.
**/   
template <typename T, typename U>
bool inline meet_data(const T& x, const U& y)
{
    if (!x.valid_device() && !y.valid_device()) { // add further tests for size 
	assert(x.valid_host() && y.valid_host());
	return true;
    }
    x.to_device(); y.to_device();
    return false;
}

/// Move all data to consistent location, preferrably the device memory
/** If data of all arguments is on host and data is not too large
    leave it on the host. In all other cases move it to the device.
    Operations to move data are supposed to be const.
    \return Whether all data reside on the host.
**/   
template <typename T, typename U, typename V>
bool inline meet_data(const T& x, const U& y, const V& z)
{
    if (!x.valid_device() && !y.valid_device() && !z.valid_device()) { // add further tests for size 
	assert(x.valid_host() && y.valid_host() && z.valid_host());
	return true;
    }
    x.to_device(); y.to_device(); z.to_device();
    return false;
}


} // namespace mtl

#endif // MTL_MEET_DATA_INCLUDE
