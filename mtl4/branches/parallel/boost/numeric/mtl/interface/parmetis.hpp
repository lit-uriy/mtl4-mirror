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

#if defined(MTL_HAS_MPI) && defined(MTL_HAS_PARMETIS)

#include <cmath>
#include <cassert>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <parmetis.h>
#include <mpiparmetis.hpp>
#include <libtopomap.hpp>
#include <boost/static_assert.hpp>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/all_to_all_sparse.hpp>
#include <boost/mpi/collectives/all_gather.hpp>

#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/par/migration.hpp>
#include <boost/numeric/mtl/par/global_non_zeros.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>

# ifdef MTL_HAS_TOPOMAP
#   include <libtopomap.hpp>
# endif


namespace mtl { namespace par {

typedef std::vector<idxtype> parmetis_index_vector;

# ifdef MTL_HAS_TOPOMAP

    void inline topology_mapping(MPI_Comm comm, parmetis_index_vector& xadj, parmetis_index_vector& adjncy, 
				 parmetis_index_vector& vtxdist, parmetis_index_vector& part)
    {
	int      numflag= 0;

        /* create communicator from Parmetis output arrays. The
         * MPIParMETIS library creates a graph communicator that
         * reflects the process neighborhoods. The Parmetis graph may
         * contain more vertices than there are ranks in the
         * communicator and depending on the mapping, there could be
         * multiple "edges" between processors. The library creates an
         * MPI topology (multi-)graph with all (multiple) edges in it.
         * This information could be used as weights */
	MPI_Comm newcomm;
	MPIX_Graph_create_parmetis_unweighted(&vtxdist[0], &xadj[0], &adjncy[0], &numflag, &part[0], &comm, &newcomm);

        /* this is all debug stuff (plots pretty cool graphs ;-)) */
	// all those filenames should somehow come from a config file (or command line)
	//TPM_Fake_names_file = (char*)"./3x3x2.fake";
	TPM_Fake_names_file = getenv("TPM_FAKE_NAMES_FILE");
        const char *ltg_output = getenv("TPM_LTG_OUTFILE");
	if(ltg_output != NULL) TPM_Write_graph_comm(newcomm, ltg_output);
        const char *ptg_output = getenv("TPM_PTG_OUTFILE");
        const char *ptg_input = getenv("TPM_PTG_INFILE");
	if(ptg_output!= NULL) TPM_Write_phystopo(newcomm, ptg_output, ptg_input);

        /* call into libToPoMap to get new permutation of ranks from
         * Parmetis output. Double edges are interpreted as weights of
         * the graph. MPI Edge weights are ignored. */
	int newrank;
	//TPM_Topomap_greedy(newcomm, "./3x3x2.graph", 0, &newrank);
        if(ptg_input == NULL) printf("MUST supply topology input file for maping (export TPM_PTG_INFILE=\"topo-file\")\n");
	TPM_Topomap(newcomm, ptg_input, 0, &newrank);

        /* Peter: der folgende Block dient nur der Veranschaulichung.
         * Die Permutation bitte dann auf die ranks (0,1,2, ... ,p) die
         * im Parmetis rauskommen anwenden */
        int p; MPI_Comm_size(newcomm, &p); // TODO: this should go away!
        int r; MPI_Comm_rank(newcomm, &r);
        std::vector<int> permutation(p);
        MPI_Allgather(&newrank, 1, MPI_INT, &permutation[0], 1, MPI_INT, newcomm);
        if(!r) { printf("rank permutation: "); for(int i=0; i<p; ++i) printf("%i ", permutation[i]); printf("\n"); }
        for(int i=0; i<part.size(); ++i) part[i] = permutation[part[i]];
    }

# endif


template <typename DistMatrix>
int parmetis_partition_k_way(const DistMatrix& A, parmetis_index_vector& xadj, parmetis_index_vector& adjncy, 
			     parmetis_index_vector& vtxdist, parmetis_index_vector& part)
{
    typedef typename DistMatrix::row_distribution_type rd_type;
    typedef typename mtl::matrix::global_non_zeros_aux<DistMatrix>::vec_type vec_type;
	
    vec_type non_zeros;
    global_non_zeros(A, non_zeros, true, false);

    // mtl::par::multiple_ostream<> mout;
    // mout << "Symmetric non-zero entries are " << non_zeros << '\n'; mout.flush();

    rd_type const&  row_dist= row_distribution(A);
    xadj.resize(num_rows(local(A))+1);
    adjncy.resize(std::max(non_zeros.size(), std::size_t(1)));
    vtxdist.resize(row_dist.size()+1);

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


template <typename DistMatrix>
int partition_k_way(const DistMatrix& A, parmetis_index_vector& part)
{
    parmetis_index_vector    xadj, adjncy, vtxdist;
    int edgecut= parmetis_partition_k_way(A, xadj, adjncy, vtxdist, part);

# ifdef MTL_HAS_TOPOMAP
    topology_mapping(communicator(row_distribution(A)), xadj, adjncy, vtxdist, part);
# endif
    
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

/// Compute a new partitioning from a matrix and create a migration upon it
template <typename DistMatrix>
block_migration inline parmetis_migration(const DistMatrix& A)
{
    parmetis_index_vector part;
    partition_k_way(A, part);

    return parmetis_migration(row_distribution(A), part);
}

}} // namespace mtl::par

#endif // MTL_HAS_PARMETI && MTL_HAS_MPI

#endif // MTL_PARMETIS_INCLUDE
