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

#ifndef MTL_PAR_MIGRATE_MATRIX_INCLUDE
#define MTL_PAR_MIGRATE_MATRIX_INCLUDE

#include <boost/numeric/mtl/matrix/inserter.hpp> 
#include <boost/numeric/mtl/matrix/traverse_distributed.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>

namespace mtl { namespace par {

    
template <typename DistMatrixA, typename DistMatrixB, typename Map> struct migrate_matrix_visitor;

/// Migrate %matrix \p A to %matrix \p B using the \p migration object and the column mapping \p new_global
template <typename DistMatrixA, typename DistMatrixB, typename Map>
void migrate_matrix(const DistMatrixA& A, DistMatrixB& B, 
		    const block_migration& migration, const Map& new_global)
{
    migrate_matrix_visitor<DistMatrixA, DistMatrixB, Map> vis(A, B, migration, new_global);
    traverse_distributed(A, vis);
}

template <typename DistMatrixA, typename DistMatrixB, typename Map> 
struct migrate_matrix_visitor
{	
    typedef typename Collection<DistMatrixA>::size_type size_type;

    migrate_matrix_visitor(const DistMatrixA& A, DistMatrixB& B, const block_migration& migration, const Map& new_global) 
      : A(A), row_dist(row_distribution(A)), col_dist(col_distribution(A)), 
	ins(B), migration(migration), new_global(new_global) 
    {}

    template <typename Matrix>
    void operator()(const Matrix& C, int p)
    {
	namespace traits= mtl::traits;
	typename traits::col<Matrix>::type             col(C); 
	typename traits::row<Matrix>::type             row(C); 
	typename traits::const_value<Matrix>::type     value(C); 
	typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
	
	for (cursor_type cursor = begin<tag::major>(C), cend = end<tag::major>(C); cursor != cend; ++cursor)
	    for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor) {
		size_type ngrow= migration.new_global(row(*icursor)), // old local to new global
		          ogcol= col_dist.local_to_global(A.decompress_column(col(*icursor), p), p), // old global
		          ngcol= new_global.find(ogcol)->second;
		ins[ngrow][ngcol] << value(*icursor);
	    }
    }

    const DistMatrixA&                                  A;
    typename DistMatrixA::row_distribution_type const&  row_dist;
    typename DistMatrixA::col_distribution_type const&  col_dist;
    mtl::matrix::inserter<DistMatrixB>                  ins;
    const block_migration&                              migration;
    const Map&                                          new_global;
};





}} // namespace mtl::par

#endif // MTL_PAR_MIGRATE_MATRIX_INCLUDE
