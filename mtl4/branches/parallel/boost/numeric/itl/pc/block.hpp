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

#ifndef ITL_PC_BLOCK_INCLUDE
#define ITL_PC_BLOCK_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>

namespace itl { namespace pc {


#ifdef MTL_HAS_MPI
/// Block preconditioner for distributed matrices, e.g. block-ILU or diagonal
template <typename DistMatrix, typename LocalPC>
class block
{
  public:
    typedef DistMatrix                                                  matrix_type;
    typedef typename mtl::Collection<matrix_type>::value_type           value_type;
    typedef typename mtl::Collection<matrix_type>::size_type            size_type;
    typedef typename mtl::DistributedCollection<DistMatrix>::local_type local_type;
    typedef block                                                       self;

    explicit block(const matrix_type& A) 
      : local_pc(local(A)), col_dist(col_distribution(A))
    {
	MTL_THROW_IF(row_distribution(A) != col_dist, mtl::incompatible_distribution());
    }

    /// Member function solve, better use free function solve
    template <typename DistVector>
    DistVector solve(const DistVector& dist_x) const
    {
	MTL_DEBUG_THROW_IF(col_dist != distribution(dist_x), mtl::incompatible_distribution());
	DistVector dist_y(dist_x); // copies distribution as well
	
	local(dist_y)= local_pc.solve(local(dist_x));
	return dist_y;
    }

    /// Member function for solving adjoint problem, better use free function adjoint_solve
    template <typename DistVector>
    DistVector adjoint_solve(const DistVector& dist_x) const
    {
	MTL_DEBUG_THROW_IF(col_dist != distribution(dist_x), mtl::incompatible_distribution());
	DistVector dist_y(dist_x); // copies distribution as well
	
	local(dist_y)= local_pc.adjoint_solve(local(dist_x));
	return dist_y;
    }

  private:
    LocalPC                                            local_pc;
    typename matrix_type::col_distribution_type const& col_dist;
};

#endif

}} // namespace itl::pc

#endif // ITL_PC_BLOCK_INCLUDE
