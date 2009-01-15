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

#ifndef MTL_DISTRIBUTION_INCLUDE
#define MTL_DISTRIBUTION_INCLUDE

#include  <boost/type_traits/is_base_of.hpp>

namespace mtl { namespace tag {

	
struct distributed {};

struct block_distributed : distributed {};

struct row_block_distributed : block_distributed {};

struct non_distributed {};

#ifdef MTL_HAS_MPI
	typedef row_block_distributed auto_distributed;
#else
	typedef non_distributed       auto_distributed;
#endif

template <typename Dist>
struct is_distributed 
    : boost::is_base_of<distributed, Dist>
{};




}} // namespace mtl::tag

#endif // MTL_DISTRIBUTION_INCLUDE
