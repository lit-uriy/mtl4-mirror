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

#if defined(MTL_HAS_PARMETIS) && defined(MTL_HAS_MPI)

#include <cmath>
#include <vector>
#include <algorithm>
#include <parmetis.h>
#include <boost/static_assert.hpp>

#include <boost/numeric/mtl/par/global_non_zeros.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>

namespace mtl { namespace matrix {

    template <typename DistMatrix>
    void partition_k_way(const DistMatrix& A)
    {
	typedef typename DistMatrix::row_distribution_type rd_type;
	typedef typename global_non_zeros_aux<DistMatrix>::vec_type vec_type;
	
	vec_type non_zeros;
	global_non_zeros(A, non_zeros, true, false);

	mtl::par::multiple_ostream<> mout;
	mout << "Symmetric non-zero entries are " << non_zeros << '\n'; mout.flush();

	rd_type const&  row_dist= row_distribution(A);
	std::vector<idxtype>    xadj(num_rows(local(A))+1), adjncy(std::max(non_zeros.size(), std::size_t(1))), vtxdist(row_dist.size()+1);

	BOOST_STATIC_ASSERT((mtl::traits::is_block_distribution<rd_type>::value));
	std::copy(row_dist.starts.begin(), row_dist.starts.end(), vtxdist.begin());
	
	int my_rank= row_dist.rank();
	unsigned xp= 0, i= 0; // position in xadj
	for (; i < non_zeros.size(); i++) {  
	    typename vec_type::value_type entry= non_zeros[i];
	    unsigned lr= row_dist.global_to_local(entry.first);
	    while (xp <= lr) xadj[xp++]= i; 
	    adjncy[i]= entry.second;
	}
	while (xp < xadj.size()) xadj[xp++]= i;
	//mout << "vtxdist = " << vtxdist << ", xadj = "    << xadj << ", adjncy = "  << adjncy << '\n';

	int                   wgtflag= 0, numflag= 0, ncon= 1, parts= row_dist.size(), options[]= {0, 0, 0}, edgecut;
	std::vector<float>    tpwgts(parts, 1.f / float(parts));
	float                 ubvec[]= {1.05};
	std::vector<idxtype>  part(adjncy.size());  
	MPI_Comm              comm(communicator(row_dist)); // 
	ParMETIS_V3_PartKway(&vtxdist[0], &xadj[0], &vtxdist[0], 0, 0, &wgtflag, &numflag, &ncon, 
			     &parts, &tpwgts[0], ubvec, options, &edgecut, &part[0], &comm);
	mout << "Edge cut = " << edgecut << ", partition = " << part << '\n';
    }
}} // namespace mtl::matrix

#endif

#endif // MTL_PARMETIS_INCLUDE
