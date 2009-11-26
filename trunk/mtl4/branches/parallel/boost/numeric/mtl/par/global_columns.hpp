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

#include <boost/numeric/mtl/operation/std_output_operator.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>
#include <boost/numeric/mtl/par/single_ostream.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>


namespace mtl { namespace matrix {


template <typename DistMatrix> struct global_columns_aux;

template <typename DistMatrix>
void global_columns(const DistMatrix& A,
		    std::vector<typename Collection<DistMatrix>::size_type>& columns)
{
    global_columns_aux<DistMatrix> g(A);
    g(columns);
}

template <typename DistMatrix> 
struct global_columns_aux
{
    typedef typename Collection<DistMatrix>::size_type size_type;
    typedef std::vector<size_type>                     vec_type;

    // typedef size_type entry_type;

    explicit global_columns_aux(const DistMatrix& A) 
      : A(A), col_dist(col_distribution(A)), my_rank(row_distribution(A).rank())
    {}

    void operator()(vec_type& columns)
    {
	// Non-zeros from local matrix
	vec_type tmp;
	extract(A.local_matrix, tmp);
	local_to_global(col_dist, tmp, my_rank); 
	eat(columns, tmp);

	// Non-zeros from remote matrices
	typedef typename DistMatrix::remote_map_type rmt;
	const rmt& remote_map(A.remote_matrices); 
	for (typename rmt::const_iterator it= remote_map.begin(), end= remote_map.end(); it != end; ++it) {
	    extract(it->second, tmp);
	    int p= it->first;

	    // decompress columns
	    const dense_vector<size_type>& index_comp= A.index_comp.find(p)->second;
	    for (unsigned i= 0, end= tmp.size(); i < end; i++)
		tmp[i]= index_comp[tmp[i]];
	    local_to_global(col_dist, tmp, p); 
	    eat(columns, tmp);
	}
    }


    template <typename Matrix>
    void extract(const Matrix& A, vec_type& v)
    {
	namespace traits= mtl::traits;
	typename traits::col<Matrix>::type             col(A); 
	typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
	
	for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor)
	    for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor)
		v.push_back(col(*icursor));
    }

    void eat(vec_type& columns, vec_type& eaten)
    {
	columns.insert(columns.end(), eaten.begin(), eaten.end());
	vec_type tmp;
	swap(eaten, tmp);
    }

  private:
    const DistMatrix& A;
    typename DistMatrix::col_distribution_type const&  col_dist;
    int             my_rank;
    // mtl::par::multiple_ostream<> mout;
};





}} // namespace mtl::matrix

#endif // MTL_MATRIX_GLOBAL_COLUMNS_INCLUDE
