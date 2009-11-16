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
#include <boost/numeric/mtl/concept/collection.hpp>

#include <boost/numeric/mtl/operation/std_output_operator.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>
#include <boost/numeric/mtl/par/single_ostream.hpp>

namespace mtl { namespace matrix {


template <typename DistMatrix> struct global_non_zeros_aux;

template <typename DistMatrix>
void global_non_zeros(const DistMatrix& A,
		      std::vector<std::pair<typename Collection<DistMatrix>::size_type,
		                            typename Collection<DistMatrix>::size_type> >& non_zeros,
		      bool symmetric= false)
{
    global_non_zeros_aux<DistMatrix> g(A);
    g(non_zeros, symmetric);
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

    void operator()(vec_type& non_zeros, bool symmetric)
    {
	// Non-zeros from local matrix
	vec_type tmp;
	extract(A.local_matrix, tmp);
	local_to_global(tmp, my_rank); 
	eat(non_zeros, tmp);

	// Non-zeros from remote matrices
	typedef typename DistMatrix::remote_map_type rmt;
	const rmt& remote_map(A.remote_matrices); 
	for (typename rmt::const_iterator it= remote_map.begin(), end= remote_map.end(); it != end; ++it) {
	    extract(it->second, tmp);
	    int p= it->first;

	    // decompress columns
	    const dense_vector<size_type>& index_comp= A.index_comp.find(p)->second;
	    for (unsigned i= 0, end= tmp.size(); i < end; i++) {
		size_type& c= tmp[i].second;
		c= index_comp[c];
	    }
	    local_to_global(tmp, p); 
	    eat(non_zeros, tmp);
	}

	if (symmetric) {
	    for (unsigned i= 0, end= non_zeros.size(); i < end; i++) 
		non_zeros.push_back(std::make_pair(non_zeros[i].second, non_zeros[i].first));
	    // Exchange and remove duplicates
	    exchange(non_zeros);
	    // mout << "Local and remote non-zeros (after exchange) " << non_zeros << '\n';
	    sort(non_zeros.begin(), non_zeros.end());
	    typename vec_type::iterator new_end = unique(non_zeros.begin(), non_zeros.end());
	    non_zeros.erase(new_end, non_zeros.end());
	    // mout << "Local and remote non-zeros (uniquely)" << non_zeros << '\n';
	}
    }

    template <typename Matrix>
    void extract(const Matrix& A, vec_type& v)
    {
	namespace traits= mtl::traits;
	typename traits::row<Matrix>::type             row(A); 
	typename traits::col<Matrix>::type             col(A); 
	typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
	
	for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor)
	    for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor)
		v.push_back(std::make_pair(row(*icursor), col(*icursor)));
    }

    void local_to_global(vec_type& non_zeros, int p)
    {
	for (unsigned i= 0; i < non_zeros.size(); i++) {
	    entry_type& nz= non_zeros[i];
	    nz.first=  row_dist.local_to_global(nz.first); 
	    nz.second= col_dist.local_to_global(nz.second, p);
	}
    }

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

	all_to_all(communicator(row_dist), send_buffers, recv_buffers);
	// mout << "Receive buffers " << recv_buffers << '\n';
	
	typename vec_type::iterator new_end = remove_if(non_zeros.begin(), non_zeros.end(), is_remote);
	non_zeros.erase(new_end, non_zeros.end());
	for (unsigned p= 0; p < row_dist.size(); p++)
	    eat(non_zeros, recv_buffers[p]);
    }

    void eat(vec_type& non_zeros, vec_type& eaten)
    {
	non_zeros.insert(non_zeros.end(), eaten.begin(), eaten.end());
	vec_type tmp;
	swap(eaten, tmp);
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

