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

#ifndef MTL_MATRIX_TRAVERSE_DISTRIBUTED_INCLUDE
#define MTL_MATRIX_TRAVERSE_DISTRIBUTED_INCLUDE

namespace mtl { namespace matrix {

    
template <typename DistMatrix, typename Visitor>
void traverse_distributed(const DistMatrix& A, Visitor& vis)
{
    typename DistMatrix::row_distribution_type const& row_dist(row_distribution(A));
    vis(A.local_matrix, row_dist.rank());

    typedef typename DistMatrix::remote_map_type rmt;
    const rmt& remote_map(A.remote_matrices); 

    for (typename rmt::const_iterator it= remote_map.begin(), end= remote_map.end(); it != end; ++it)
	vis(it->second, it->first);
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_TRAVERSE_DISTRIBUTED_INCLUDE
