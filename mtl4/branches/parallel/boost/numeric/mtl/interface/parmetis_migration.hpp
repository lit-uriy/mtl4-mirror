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

#ifndef MTL_PAR_PARMETIS_MIGRATION_INCLUDE
#define MTL_PAR_PARMETIS_MIGRATION_INCLUDE

#ifdef MTL_HAS_MPI // needs MPI but not necessarily ParMetis

#include <vector>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/all_to_all_sparse.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/par/migration.hpp>


namespace mtl { namespace par {

// Maybe pre-compile
template <typename Partitioning>
block_migration inline parmetis_migration(const block_distribution& old_dist, const Partitioning& part)
{
    using std::size_t;
    // mtl::par::multiple_ostream<> mout;

    block_migration migration(old_dist);

    // Telling new owner which global indices I'll give him
    std::vector<std::vector<size_t> > send_glob(old_dist.size()), recv_glob;
    for (unsigned i= 0; i < part.size(); i++) 
	send_glob[part[i]].push_back(old_dist.local_to_global(i));
    boost::mpi::communicator comm(communicator(old_dist));
    all_to_all_sparse(comm, send_glob, recv_glob);
    // mout << "Sended " << send_glob << "\nReceived " << recv_glob << '\n';
    { std::vector<std::vector<size_t> > tmp(comm.size()); swap(tmp, send_glob); } // release memory


    // Which global indices in the old dist are my local indices in the new distribution
    for (size_t p= 0; p < recv_glob.size(); p++) {
	const std::vector<size_t>& from_p= recv_glob[p];
	for (size_t i= 0; i < from_p.size(); i++)
	    migration.add_old_global(from_p[i]);
    }
    // mout << "new_to_old is " << migration.new_to_old << '\n';
    { std::vector<std::vector<size_t> > tmp; swap(tmp, recv_glob); } // release memory
    
    // Build new distribution
    size_t              my_size= migration.new_local_size();
    std::vector<size_t> all_sizes;
    all_gather(comm, my_size, all_sizes);

    migration.new_dist.setup_from_local_sizes(all_sizes);
    // mout << "New distribution is " << new_dist.starts << '\n';

    // Send the new global indices back to the owner in the old distribution (to perform migration)
    for (size_t i= 0; i < my_size; i++)
	send_glob[old_dist.on_rank(migration.old_global(i))].push_back(migration.new_dist.local_to_global(i));
    all_to_all_sparse(comm, send_glob, recv_glob);
    // mout << "Sended " << send_glob << "\nReceived " << recv_glob << '\n';
    { std::vector<std::vector<size_t> > tmp(comm.size()); swap(tmp, send_glob); } // release memory
    
    // Build old to new mapping; relies on keeping relative orders of indices
    std::vector<size_t> counters(comm.size(), 0);
    for (size_t i= 0; i < part.size(); i++) {
	size_t p= part[i];
	migration.add_new_global(recv_glob[p][counters[p]++]);
    }
    // mout << "old_to_new is " << migration.old_to_new << '\n';

    return migration;
}

}} // namespace mtl::par

#endif //  MTL_HAS_MPI

#endif // MTL_PAR_PARMETIS_MIGRATION_INCLUDE
