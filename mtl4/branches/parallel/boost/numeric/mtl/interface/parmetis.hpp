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

#ifndef MTL_PARMETIS_INCLUDE
#define MTL_PARMETIS_INCLUDE

#if defined(MTL_HAS_MPI)

#if defined(MTL_HAS_PARMETIS)

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <parmetis.h>
#include <boost/static_assert.hpp>

#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/par/migration.hpp>
#include <boost/numeric/mtl/par/global_non_zeros.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>

namespace mtl { namespace par {


typedef std::vector<idxtype> parmetis_index_vector;

template <typename DistMatrix>
int partition_k_way(const DistMatrix& A, parmetis_index_vector& part)
{
    typedef typename DistMatrix::row_distribution_type rd_type;
    typedef typename mtl::matrix::global_non_zeros_aux<DistMatrix>::vec_type vec_type;
	
    vec_type non_zeros;
    global_non_zeros(A, non_zeros, true, false);

    // mtl::par::multiple_ostream<> mout;
    // mout << "Symmetric non-zero entries are " << non_zeros << '\n'; mout.flush();

    rd_type const&  row_dist= row_distribution(A);
    parmetis_index_vector    xadj(num_rows(local(A))+1), adjncy(std::max(non_zeros.size(), std::size_t(1))), vtxdist(row_dist.size()+1);

    BOOST_STATIC_ASSERT((mtl::traits::is_block_distribution<rd_type>::value));
    std::copy(row_dist.starts.begin(), row_dist.starts.end(), vtxdist.begin());
	
    int my_rank= row_dist.rank();
    unsigned i= 0, xp= 0; // position in xadj
    for (; i < non_zeros.size(); i++) {  
	typename vec_type::value_type entry= non_zeros[i];
	unsigned lr= row_dist.global_to_local(entry.first);
	while (xp <= lr) xadj[xp++]= i; 
	adjncy[i]= entry.second;
    }
    while (xp < xadj.size()) xadj[xp++]= i;
    // mout << "vtxdist = " << vtxdist << ", xadj = "    << xadj << ", adjncy = "  << adjncy << '\n';

    int                   wgtflag= 0, numflag= 0, ncon= 0, nparts= row_dist.size(), options[]= {0, 0, 0}, edgecut;
    idxtype               *vwgt= 0, *adjwgt= 0;
    float                 *tpwgts= 0, *ubvec= 0;
    // std::vector<float>    tpwgts(parts, 1.f / float(parts)); // ignored right now
    // float                 ubvec[]= {1.05};                   // ignored right now
    MPI_Comm              comm(communicator(row_dist)); // 

    part.resize(xadj.size()); 
    ParMETIS_V3_PartKway(&vtxdist[0], &xadj[0], &adjncy[0], vwgt, adjwgt, &wgtflag, &numflag, &ncon, 
			 &nparts, tpwgts, ubvec, options, &edgecut, &part[0], &comm);
    part.pop_back(); // to avoid empty vector part has extra entry
    // mout << "Edge cut = " << edgecut << ", partition = " << part << '\n';
    return edgecut;
}


// Maybe pre-compile
block_migration inline parmetis_migration(const block_distribution& old_dist, const parmetis_index_vector& part)
{
    using std::size_t;
    mtl::par::multiple_ostream<> mout;

    block_migration migration(old_dist);

    // Telling new owner which global indices I'll give him
    std::vector<std::vector<size_t> > send_glob(old_dist.size()), recv_glob;
    for (unsigned i= 0; i < part.size(); i++) 
	send_glob[part[i]].push_back(old_dist.local_to_global(i));
    boost::mpi::communicator comm(communicator(old_dist));
    all_to_all(comm, send_glob, recv_glob);
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
    all_to_all(comm, send_glob, recv_glob);
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

/// Compute a new partitioning from a matrix and create a migration upon it
template <typename DistMatrix>
block_migration inline parmetis_migration(const DistMatrix& A)
{
    std::vector<idxtype> part;
    partition_k_way(A, part);

    return parmetis_migration(row_distribution(A), part);
}

}} // namespace mtl::par

#else // MTL_HAS_PARMETIS

// To make more tests parmetis-independent
typedef long int idxtype;

#endif // MTL_HAS_PARMETIS

#endif // MTL_HAS_MPI

#endif // MTL_PARMETIS_INCLUDE
