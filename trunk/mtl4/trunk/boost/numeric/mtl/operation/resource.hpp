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

#ifndef MTL_RESOURCE_INCLUDE
#define MTL_RESOURCE_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl {

    namespace vector {

	/// Describes the resources need for a certain vector.
	/** All necessary information to construct appropriate/consistent temporary vectors.
	    Normally, this is just the size of the vector.
	    For distributed vector we also need its distribution. **/
	template <typename Vector>
	typename Collection<Vector>::size_type inline resource(const Vector& v)
	{
	    return size(v);
	}

    } // namespace vector

    namespace matrix {
	// maybe a pair of size_type? like position
    }

} // namespace mtl

#endif // MTL_RESOURCE_INCLUDE
