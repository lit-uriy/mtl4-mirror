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

#ifndef MTL_PAR_MIGRATE_VECTOR_INCLUDE
#define MTL_PAR_MIGRATE_VECTOR_INCLUDE

#ifdef MTL_HAS_MPI

#include <boost/numeric/mtl/concept/collection.hpp> 
#include <boost/numeric/mtl/vector/inserter.hpp> 
#include <boost/numeric/mtl/operation/local.hpp> 
#include <boost/numeric/mtl/par/migration.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>

namespace mtl { namespace par {

/// Migrate %vector \p v to %vector \p w using the \p migration object 
template <typename DistVectorV, typename DistVectorW>
void migrate_vector(const DistVectorV& v, DistVectorW& w, const block_migration& migration)
{
    typedef typename DistributedCollection<DistVectorV>::local_type local_type;
    typedef typename Collection<local_type>::size_type              size_type;
    
    set_to_zero(local(w)); // to avoid undefined entries
    vector::inserter<DistVectorW> ins(w);

    const local_type& l= local(v);
    for (size_type i= 0; i < size(l); ++i)
	ins[migration.new_global(i)] << l[i];
}

}} // namespace mtl::par

#endif // MTL_HAS_MPI


#endif // MTL_PAR_MIGRATE_VECTOR_INCLUDE
