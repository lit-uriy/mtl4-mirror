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

#include <utility>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

#ifdef MTL_HAS_MPI
#  include <boost/numeric/mtl/vector/distributed.hpp>
#endif 

namespace mtl {

    namespace traits {

	template <typename Vector>
	struct vector_resource
	{
	    typedef typename Collection<Vector>::size_type type;
	    type inline static apply(const Vector& v) { using mtl::vector::size; return size(v); }
	};

#     ifdef MTL_HAS_MPI
	template <typename Vector, typename Distribution>
	struct vector_resource< vector::distributed<Vector, Distribution> >
	{
	    typedef vector::distributed<Vector, Distribution> arg_type;
	    typedef typename Collection<arg_type>::size_type  size_type;
	    typedef std::pair<size_type, Distribution>        type;
	    type inline static apply(const arg_type& v)
	    {
		using mtl::vector::size; using mtl::vector::distribution;
		return std::make_pair(size(v), distribution(v));
	    }
	};
#     endif
    }

    namespace vector {

	/// Describes the resources need for a certain vector.
	/** All necessary information to construct appropriate/consistent temporary vectors.
	    Normally, this is just the size of the vector.
	    For distributed vector we also need its distribution. **/
	template <typename Vector>
	typename mtl::traits::vector_resource<Vector>::type
	inline resource(const Vector& v)
	{
	    return mtl::traits::vector_resource<Vector>::apply(v);
	}

    } // namespace vector

    namespace matrix {
	// maybe a pair of size_type? like position
    }

} // namespace mtl

#endif // MTL_RESOURCE_INCLUDE
