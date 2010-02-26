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

#ifndef MTL_MIGRATING_COPY_INCLUDE
#define MTL_MIGRATING_COPY_INCLUDE

#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp> 
#include <boost/numeric/mtl/matrix/traverse_distributed.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/copy_inserter_size.hpp>

namespace mtl {

    namespace matrix {

	template <typename Updater, typename DistMatrixA, typename DistMatrixB, bool Transposed= false> 
	struct migrating_copy_visitor
	{
	    typedef typename Collection<DistMatrixA>::size_type size_type;

	    migrating_copy_visitor(const DistMatrixA& A, DistMatrixB& B) 
	      : A(A), row_dist(row_distribution(A)), col_dist(col_distribution(A)), 
		ins(B, mtl::detail::copy_inserter_size<Updater>::apply(A, B))
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
			size_type grow= row_dist.local_to_global(row(*icursor)), 
			          gcol= col_dist.local_to_global(A.decompress_column(col(*icursor), p), p);
			if (Transposed)
			    ins[gcol][grow] << value(*icursor);
			else
			    ins[grow][gcol] << value(*icursor);
		    }
	    }

	  private:
	    const DistMatrixA&                                  A;
	    typename DistMatrixA::row_distribution_type const&  row_dist;
	    typename DistMatrixA::col_distribution_type const&  col_dist;
	    mtl::matrix::inserter<DistMatrixB, Updater>         ins;
	};

	/// Copy matrix \p A into matrix \p B where \p A and \p B might have different distributions but same global indexing.
	template <typename Updater, typename DistMatrixA, typename DistMatrixB>
	inline void migrating_copy(const DistMatrixA& A, DistMatrixB& B)
	{
	    if (Updater::init_to_zero)
		set_to_zero(B);
	    migrating_copy_visitor<Updater, DistMatrixA, DistMatrixB> vis(A, B);
	    traverse_distributed(A, vis);
	}	

	/// Copy matrix \p A into matrix \p B where \p A and \p B might have different distributions but same global indexing.
	template <typename DistMatrixA, typename DistMatrixB>
	inline void migrating_copy(const DistMatrixA& A, DistMatrixB& B)
	{
	    typedef operations::update_store<typename Collection<DistMatrixB>::value_type> Updater;
	    migrating_copy<Updater, DistMatrixA, DistMatrixB>(A, B);
	}
	
	/// Copy transposed view \p A into matrix \p B where \p A and \p B might have different distributions but same global indexing.
	template <typename Updater, typename DistMatrixA, typename DistMatrixB>
	inline void transposed_migrating_copy(const DistMatrixA& A, DistMatrixB& B)
	{
	    if (Updater::init_to_zero)
		set_to_zero(B);
	    migrating_copy_visitor<Updater, typename DistMatrixA::other, DistMatrixB, true> vis(A.ref, B);
	    traverse_distributed(A.ref, vis);
	}	

	/// Copy transposed view  \p A into matrix \p B where \p A and \p B might have different distributions but same global indexing.
	template <typename DistMatrixA, typename DistMatrixB>
	inline void transposed_migrating_copy(const DistMatrixA& A, DistMatrixB& B)
	{
	    typedef operations::update_store<typename Collection<DistMatrixB>::value_type> Updater;
	    transposed_migrating_copy<Updater, DistMatrixA, DistMatrixB>(A, B);
	}	
    }

    namespace vector {} // do we need this?


} // namespace mtl

#endif // MTL_MIGRATING_COPY_INCLUDE
