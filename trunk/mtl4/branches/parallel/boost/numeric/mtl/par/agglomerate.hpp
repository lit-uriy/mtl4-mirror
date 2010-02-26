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

#ifndef MTL_AGGLOMERATE_INCLUDE
#define MTL_AGGLOMERATE_INCLUDE

#ifdef MTL_HAS_MPI 

#include <vector>
#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/par/migrating_copy.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl {

    namespace matrix {

	/// Agglomerate distributed matrix \p A on \p rank (default is 0) by migrating into matrix of type \p Matrix.
	/** Result type is local type of \p Matrix. Other processors return empty matrix. 
	    Row and column distribution cannot be (block) cyclic because all entries go to one rank. **/
	template <typename Matrix, typename Src>
	typename mtl::DistributedCollection<Matrix>::local_type
	inline agglomerate(const Src& A, std::size_t rank= 0)
	{
	    typedef typename mtl::DistributedCollection<Matrix>::local_type local_type;

	    int csize= communicator(A).size();
	    assert(rank >= 0 && rank < csize);

	    std::vector<std::size_t> row_blocks(csize+1, 0), col_blocks(csize+1, 0);
	    for (std::size_t i= rank+1; i <= csize; i++) 
		row_blocks[i]= num_rows(A), col_blocks[i]= num_cols(A);
	    par::block_distribution row_dist(row_blocks), col_dist(col_blocks);
	    Matrix B(num_rows(A), num_cols(A), row_dist, col_dist);
	    migrating_copy(A, B);
#if 0
	    typedef typename mtl::DistributedCollection<Matrix>::local_type local_type;
	    std::vector<std::size_t> part(max(num_rows(local(A)), num_cols(local(A))), rank);
	    Matrix B(A, par::parmetis_migration(row_distribution(A), part));
#endif
	    return communicator(A).rank() == rank ? local(B) : local_type();
	}

	/// Agglomerate distributed matrix \p A on \p rank (default is 0) by migration.
	/** Result type is local type of \p A. Other processors return empty matrix. **/
	template <typename Src>
	typename mtl::DistributedCollection<Src>::local_type
	inline agglomerate(const Src& A, std::size_t rank= 0)
	{
	    return agglomerate<Src, Src>(A, rank);
	}
    }

    namespace vector {

	/// Agglomerate distributed vector \p A on \p rank (default is 0)
	/** Type is local type of \p A. Other processors return empty vector. **/
	template <typename Vector>
	typename mtl::DistributedCollection<Vector>::local_type
	inline agglomerate(const Vector& v, std::size_t rank= 0)
	{
	    typedef typename mtl::DistributedCollection<Vector>::local_type local_type;
	    std::vector<std::size_t> part(num_rows(local(v)), rank);
	    Vector w(v, parmetis_migration(row_distribution(v), part));
	    return communicator(v).rank() == rank ? local(w) : local_type();
	}
    }

} // namespace mtl

#endif //  MTL_HAS_MPI

#endif // MTL_AGGLOMERATE_INCLUDE
