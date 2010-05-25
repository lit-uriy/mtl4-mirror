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
#include <boost/static_assert.hpp>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/all_to_all_sparse.hpp>
#include <boost/mpi/collectives/all_gather.hpp>

#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/par/migration.hpp>
#include <boost/numeric/mtl/par/global_non_zeros.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>
#include <boost/numeric/mtl/interface/parmetis_migration.hpp>

# ifdef MTL_HAS_TOPOMAP
#   include <mpiparmetis.hpp>
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

#       if 1 // ndef NDEBUG // oder #ifdef TMP_PLOT_GRAPHS
          /* this is all debug stuff (plots pretty cool graphs ;-)) */
	  // all those filenames should somehow come from a config file (or command line)
  	  //TPM_Fake_names_file = (char*)"./3x3x2.fake";
	  TPM_Fake_names_file = getenv("TPM_FAKE_NAMES_FILE"); // Wo kommt das her??? Globale Variable im globalem Namensraum?
	  const char *ltg_output = getenv("TPM_LTG_OUTFILE"), *ptg_output = getenv("TPM_PTG_OUTFILE"),
	             *ptg_input = getenv("TPM_PTG_INFILE");
	  if (ltg_output != NULL) TPM_Write_graph_comm(newcomm, ltg_output); // den Nulltest kannst Du auch in der Funktion machen (einmal)
	  if (ptg_output!= NULL) TPM_Write_phystopo(newcomm, ptg_output, ptg_input);
#       endif

        /* call into libtopomap to get new permutation of ranks from
         * Parmetis output. Double edges are interpreted as weights of
         * the graph. MPI Edge weights are ignored. */
	int newrank;
        if(ptg_input == NULL) printf("MUST supply topology input file for maping (export TPM_PTG_INFILE=\"topo-file\")\n");
	TPM_Topomap(newcomm, ptg_input, 0, &newrank);

        int p; MPI_Comm_size(newcomm, &p); // TODO: this should go away!
        std::vector<int> permutation(p);
        MPI_Allgather(&newrank, 1, MPI_INT, &permutation[0], 1, MPI_INT, newcomm);

#       if 1 // ndef NDEBUG 
        int r; MPI_Comm_rank(newcomm, &r);
        if(!r) { printf("rank permutation: "); for(int i=0; i<p; ++i) printf("%i ", permutation[i]); printf("\n"); }
#       endif

        //std::vector<int> rperm(p); // reverse permutation
        //for(int i=0; i<p; ++i) rperm[permutation[i]] = i;

        // rank r should behave like it was rank permutation[r]!
        for(int i=0; i<part.size(); ++i) part[i] = permutation[part[i]];

        // benchmark topology with simple data transmission!
        double t1,t2;
        TPM_Benchmark_graphtopo(newcomm, newrank, 1024 /*dsize*/, &t1, &t2);
        if(!r) printf("topo benchmark: %f s -> %f s\n", t1, t2);
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

/// Compute a new partitioning from a matrix and create a migration upon it
template <typename DistMatrix>
block_migration inline parmetis_migration(const DistMatrix& A)
{
    parmetis_index_vector part;
    partition_k_way(A, part);

    return parmetis_migration(row_distribution(A), part);
}


}} // namespace mtl::par

#endif // MTL_HAS_PARMETIS && MTL_HAS_MPI

#endif // MTL_PARMETIS_INCLUDE
