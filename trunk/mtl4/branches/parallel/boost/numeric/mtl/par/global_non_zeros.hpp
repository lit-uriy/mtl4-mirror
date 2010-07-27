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

#ifndef MTL_MATRIX_GLOBAL_NON_ZEROS_INCLUDE
#define MTL_MATRIX_GLOBAL_NON_ZEROS_INCLUDE

#include <utility>
#include <vector>
#include <algorithm>
#include <boost/mpi/collectives/all_to_all_sparse.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/stl_extension.hpp>
#include <boost/numeric/mtl/matrix/traverse_distributed.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>

#include <boost/numeric/mtl/operation/std_output_operator.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>
#include <boost/numeric/mtl/par/single_ostream.hpp>

namespace mtl { namespace matrix {


template <typename DistMatrix> struct global_non_zeros_aux;

template <typename DistMatrix>
void global_non_zeros(const DistMatrix& A,
		      std::vector<std::pair<typename Collection<DistMatrix>::size_type,
		                            typename Collection<DistMatrix>::size_type> >& non_zeros,
		      bool symmetric= false,
		      bool with_diagonal= true)
{
    global_non_zeros_aux<DistMatrix> g(A);
    g(non_zeros, symmetric, with_diagonal);
}

template <typename DistMatrix> 
struct global_non_zeros_aux
{
    typedef typename Collection<DistMatrix>::size_type size_type;
    typedef std::pair<size_type, size_type>            entry_type;
    typedef std::vector<entry_type>                    vec_type;
    typedef typename DistMatrix::row_distribution_type rd_type;
    typedef typename DistMatrix::col_distribution_type cd_type;

    explicit global_non_zeros_aux(const DistMatrix& A) 
      : A(A), row_dist(row_distribution(A)), col_dist(col_distribution(A)), my_rank(row_dist.rank())
    {}

    struct is_reflexive_t {
	bool operator()(const entry_type& nz) { return nz.first == nz.second; }
    };

    struct matrix_visitor;

    void operator()(vec_type& non_zeros, bool symmetric, bool with_diagonal)
    {
	matrix_visitor vis(A, non_zeros);
	traverse_distributed(A, vis);
	
	if (!with_diagonal) {
	    typename vec_type::iterator new_end = remove_if(non_zeros.begin(), non_zeros.end(), is_reflexive_t());
	    non_zeros.erase(new_end, non_zeros.end());
	}

	if (symmetric) {
	    for (unsigned i= 0, end= non_zeros.size(); i < end; i++) 
		non_zeros.push_back(std::make_pair(non_zeros[i].second, non_zeros[i].first));
	    // Exchange and remove duplicates
	    exchange(non_zeros);
	    // mout << "Local and remote non-zeros (after exchange) " << non_zeros << '\n';
	    only_unique(non_zeros);
	    // mout << "Local and remote non-zeros (uniquely)" << non_zeros << '\n';
	}
    }

    struct matrix_visitor
    {	
	explicit matrix_visitor(const DistMatrix& D, vec_type& non_zeros) 
	  : D(D), non_zeros(non_zeros) {}

	template <typename Matrix>
	void operator()(const Matrix& A, int p)
	{
	    vec_type tmp;
	    extract(A, tmp, p);
	    local_to_global(tmp, p);
	    consume(non_zeros, tmp);
	}

	template <typename Matrix>
	void extract(const Matrix& A, vec_type& v, int p)
	{
	    namespace traits= mtl::traits;
	    typename traits::row<Matrix>::type             row(A); 
	    typename traits::col<Matrix>::type             col(A); 
	    typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	    typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
	
	    for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor)
		for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor)
		    v.push_back(std::make_pair(row(*icursor), D.decompress_column(col(*icursor), p)));
	}

	void local_to_global(vec_type& non_zeros, int p)
	{
	    typename DistMatrix::row_distribution_type const& row_dist(row_distribution(D));
	    typename DistMatrix::row_distribution_type const& col_dist(col_distribution(D));
	    for (unsigned i= 0; i < non_zeros.size(); i++) {
		entry_type& nz= non_zeros[i];
		nz.first=  row_dist.local_to_global(nz.first); 
		nz.second= col_dist.local_to_global(nz.second, p);
	    }
	}

	const DistMatrix& D;
	vec_type& non_zeros;
    };

    struct is_remote_t
    {
	is_remote_t(const rd_type& rd) : row_dist(rd) {}
	bool operator()(const entry_type& nz) { return !row_dist.is_local(nz.first);	}
	rd_type const&  row_dist;
    };

    void exchange(vec_type& non_zeros)
    {
	is_remote_t is_remote(row_dist);
	std::vector<vec_type>   send_buffers(row_dist.size()), recv_buffers(row_dist.size());

	for (unsigned i= 0; i < non_zeros.size(); i++) {
	    entry_type& nz= non_zeros[i];
	    if (is_remote(nz)) 
		send_buffers[row_dist.on_rank(nz.first)].push_back(nz);
	}
	// mout << "Send buffers " << send_buffers << '\n';

	all_to_all_sparse(communicator(row_dist), send_buffers, recv_buffers);
	// mout << "Receive buffers " << recv_buffers << '\n';
	
	typename vec_type::iterator new_end = remove_if(non_zeros.begin(), non_zeros.end(), is_remote);
	non_zeros.erase(new_end, non_zeros.end());
	for (int p= 0; p < row_dist.size(); p++)
	    consume(non_zeros, recv_buffers[p]);
    }

  private:
    const DistMatrix& A;
    rd_type const&  row_dist;
    cd_type const&  col_dist;
    int             my_rank;
    // mtl::par::multiple_ostream<> mout;
};



}} // namespace mtl::matrix

#endif // MTL_MATRIX_GLOBAL_NON_ZEROS_INCLUDE






#if 0 // might be faster for exchange
	std::vector<int>        have_msg(row_dist.size(), 0);
	std::vector<vec_type*>  remote_non_zeros(row_dist.size(), 0);
	for (unsigned i= 0; i < non_zeros.size(); i++) {
	    entry_type& nz= non_zeros[i];
	    if (is_remote(nz)) {
		int p= row_dist.on_rank(nz.first);
		have_msg[p]= 1;
		if (!remote_non_zeros[p])
		    remote_non_zeros[p]= new vec_type;
		remote_non_zeros[p]->push_back(nz);
	    }
	}
#endif

