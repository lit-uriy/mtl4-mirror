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

#include <vector>

namespace mtl { namespace par {

// More general implementation (between arbitrary distributions) will follow
template <typename OldDist, typename NewDist> class migration {};

/// Class for handling the migration from one block distribution to another
class block_migration 
{
    /// Size type
    typedef std::size_t     size_type;

  public:
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

    //private:
    std::vector<size_type> old_to_new, new_to_old;
};


}} // namespace mtl::par

#endif // MTL_PAR_MIGRATION_INCLUDE
