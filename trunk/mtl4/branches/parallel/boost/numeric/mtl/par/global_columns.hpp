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

#ifndef MTL_MATRIX_GLOBAL_COLUMNS_INCLUDE
#define MTL_MATRIX_GLOBAL_COLUMNS_INCLUDE

#include <utility>
#include <vector>
#include <algorithm>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/stl_extension.hpp>
#include <boost/numeric/mtl/matrix/traverse_distributed.hpp>

#include <boost/numeric/mtl/operation/std_output_operator.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>
#include <boost/numeric/mtl/par/single_ostream.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>


namespace mtl { namespace matrix {


template <typename DistMatrix> struct global_columns_visitor;

template <typename DistMatrix>
void global_columns(const DistMatrix& D,
		    std::vector<typename Collection<DistMatrix>::size_type>& columns)
{
    global_columns_visitor<DistMatrix> vis(D, columns);
    traverse_distributed(D, vis);
    sort(columns.begin(), columns.end());
}


template <typename DistMatrix> 
struct global_columns_visitor
{	
    typedef typename Collection<DistMatrix>::size_type size_type;
    typedef std::vector<size_type>                     vec_type;

    explicit global_columns_visitor(const DistMatrix& D, vec_type& columns) 
      : D(D), col_dist(col_distribution(D)), columns(columns) {}

    template <typename Matrix>
    void operator()(const Matrix& A, int p)
    {
	namespace traits= mtl::traits;
	typename traits::col<Matrix>::type             col(A); 
	typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
	
	vec_type tmp;
	for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor)
	    for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor)
		tmp.push_back(D.decompress_column(col(*icursor), p));
	// std::cout << "On " << col_dist.rank() << ": columns (local indices) are " << tmp << " w.r.t. processor " << p << std::endl;
	local_to_global(col_dist, tmp, p); 
	only_unique(tmp);
	// std::cout << "On " << col_dist.rank() << ": columns (global indices) are " << tmp << std::endl;
	consume(columns, tmp);
	// std::cout << "On " << col_dist.rank() << ": all columns (global indices) are " << columns << std::endl;
    }

    const DistMatrix& D;
    typename DistMatrix::col_distribution_type const&  col_dist;
    vec_type& columns;
};
    
}} // namespace mtl::matrix

#endif // MTL_MATRIX_GLOBAL_COLUMNS_INCLUDE
