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

#ifndef MTL_MATRIX_DISTRIBUTED_INCLUDE
#define MTL_MATRIX_DISTRIBUTED_INCLUDE

#include <boost/numeric/mtl/parallel/distribution.hpp>

namespace mtl { namespace matrix {


template <typename Matrix, typename Distribution = par::block_row_distribution>
class distributed
{
    typedef typename Collection<Matrix>::size_type size_type;
public:
    explicit distributed(size_type grows, size_type gcols, 
			 const Distribution& dist= Distribution(grows, gcols)) 
	: dist(dist), raw_local(dist.local_num_rows(grows), dist.local_num_cols(gcols)) {}

    boost::mpi::communicator communicator() const { return dist.communicator(); }

    Distribution      dist;
    Matrix            raw_local;
}



}} // namespace mtl::matrix

#endif // MTL_MATRIX_DISTRIBUTED_INCLUDE
