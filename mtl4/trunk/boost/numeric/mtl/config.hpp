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

#ifndef MTL_CONFIG_INCLUDE
#define MTL_CONFIG_INCLUDE

namespace mtl {


    namespace matrix {

#ifdef MTL_MATRIX_COMPRESSED_LINEAR_SEARCH_LIMIT
	/// Maximal number of entries that is searched linearly; above this std::lower_bound is used.
	const unsigned compressed_linear_search_limit= MTL_MATRIX_COMPRESSED_LINEAR_SEARCH_LIMIT;
#else
	const unsigned compressed_linear_search_limit= 10;
#endif

    }



} // namespace mtl

#endif // MTL_CONFIG_INCLUDE
