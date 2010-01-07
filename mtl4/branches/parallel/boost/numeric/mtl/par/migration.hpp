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

#ifndef MTL_PAR_MIGRATION_INCLUDE
#define MTL_PAR_MIGRATION_INCLUDE

#ifdef MTL_HAS_MPI

#include <vector>
#include <boost/numeric/mtl/par/distribution.hpp>

#ifdef MTL_HAS_PARMETIS
#  include <parmetis.h>
#endif

namespace mtl { namespace par {

// More general implementation (between arbitrary distributions) will follow
template <typename OldDist, typename NewDist> class migration {};

/// Class for handling the migration from one block distribution to another
class block_migration 
{
  public:
    /// Size type
    typedef base_distribution::size_type     size_type;

    /// Constructor refers establishes references to old distribution (new distribution remains empty so far)
    block_migration(const block_distribution& old_dist) 
      : old_dist(old_dist), new_dist(old_dist.size(), communicator(old_dist)) {}

    /// Constructor refers establishes references to old distribution and copies new one
    block_migration(const block_distribution& old_dist, const block_distribution& new_dist) 
      : old_dist(old_dist), new_dist(new_dist) {}

    /// Compute the global index in the new distribution from the local index in the old one
    /** Other indices are not known. It has to be computed by the according process **/
    size_type new_global(size_type old_local) const { return old_to_new[old_local]; }

    /// Compute the global index in the old distribution from the local index in the new one
    /** Other indices are not known. It has to be computed by the according process **/
    size_type old_global(size_type new_local) const { return new_to_old[new_local]; }

    /// Add a global old index to the local new distribution
    /** Needed to set up a new block distribution. **/
    void add_old_global(size_type old_global) { new_to_old.push_back(old_global); }

    /// Add a global new index to the local old distribution
    /** Needed to compute new global index. **/
    void add_new_global(size_type new_global) { old_to_new.push_back(new_global); }

    /// Size of local partition in new distribution 
    /** Needed to set up a new block distribution. **/
    size_type new_local_size() const { return new_to_old.size(); }

    /// Reference to new distribution
    const block_distribution& new_distribution() const { return new_dist; }

    /// Reference to old distribution
    const block_distribution& old_distribution() const { return old_dist; }

# ifdef MTL_HAS_PARMETIS
    friend block_migration parmetis_migration(const block_distribution&, const std::vector<idxtype>&);
# endif
    friend block_migration inline reverse(const block_migration& src);
    template <typename Indices, typename Map> friend void new_global_map(const block_migration&, const Indices&, Map&);

  private:
    std::vector<size_type> old_to_new, new_to_old;
    const block_distribution &old_dist;
    block_distribution       new_dist;
};

/// Reverse migration: roles of old and new distribution are interchanged
/** A view would be more efficient but would take more meta-programming.
    Extension to migrations between arbitrary distributions on demand. **/
block_migration inline reverse(const block_migration& src)
{
    block_migration rev(src.new_distribution(), src.old_distribution());
    rev.old_to_new= src.new_to_old;
    rev.new_to_old= src.old_to_new;
    return rev;
}

/// Migration object that agglomerates everything on rank 0
block_migration inline agglomerated_migration(const block_distribution& src)
{
    typedef block_migration::size_type size_type;
    size_type n= src.max_global(), nl= src.num_local(n);
    std::vector<size_type> vec(src.size()+1, n);
    vec[0]= 0;
    block_migration migr(src, block_distribution(vec, communicator(src)));

    // Global indices do not change
    if (src.rank() == 0)
	for (size_type i= 0; i < n; i++)
	    migr.add_old_global(i);
    for (size_type i= 0; i < nl; i++)
	migr.add_new_global(src.local_to_global(i));
    
    return migr;
}

}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_MIGRATION_INCLUDE
