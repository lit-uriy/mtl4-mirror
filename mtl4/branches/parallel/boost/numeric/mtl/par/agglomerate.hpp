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

#ifndef MTL_AGGLOMERATE_INCLUDE
#define MTL_AGGLOMERATE_INCLUDE

namespace mtl {

    namespace matrix {

	template <typename Matrix>
	typename mtl::DistributedCollection<Matrix>::local_type
	inline agglomerate(const Matrix& A)
	{
	    typedef typename mtl::Collection<Matrix>::size_type size_type;
	    


	}
    }


} // namespace mtl

#endif // MTL_AGGLOMERATE_INCLUDE
