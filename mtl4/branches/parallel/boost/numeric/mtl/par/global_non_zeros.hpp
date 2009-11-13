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

#include <vector>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl { namespace matrix {


template <typename DistMatrix> struct global_non_zeros_aux;

template <typename DistMatrix>
void global_non_zeros(const DistMatrix& A,
		      std::vector<std::pair<typename Collection<Matrix>::size_type>,
		                            typename Collection<Matrix>::size_type> > >& non_zeros,
		      bool symmetric= false)
{
    global_non_zeros_aux<DistMatrix>(A)(non_zeros, symmetric);
}

template <typename DistMatrix> 
struct global_non_zeros_aux
{
    typedef typename Collection<Matrix>::size_type     size_type;
    typedef std::pair<size_type, size_type>            entry_type;
    typedef std::vector<entry_type>                    vec_type;
    typedef typename DistMatrix::row_distribution_type rd_type;

    explicit global_non_zeros_aux(const DistMatrix& A) 
      : A(A), row_dist(row_distribution(A)) {}

    void operator()(vec_type& non_zeros, bool symmetric)
    {
	// Non-zeros from local matrix
	vec_type tmp;
	extract(A.local, tmp);
	size_type col_shift= A.cdp->local_to_global(0);
	local_to_global(tmp, row_dist.local_to_global(0), col_shift);
	eat(non_zeros, tmp);
	
	// Non-zeros from remote matrices
	typedef typename DistMatrix::remote_map_type rmt;
	const rmt& remote_map(A.remote_matrices); 
	for (typename rmt::const_iterator it= remote_map.begin(), end= remote_map.end(); it != end; ++it) {
	    extract(it->second, tmp);
	    int p= it->first;
	    local_to_global(tmp, row_dist.local_to_global(0, p), col_shift);
	    eat(non_zeros, tmp);
	}

	if (symmetric) {
	    for (unsigned i= 0, end= non_zeros.size(); i < end; i++) 
		non_zeros.push_back(std::make_pair(non_zeros[i].second, non_zeros[i].first));
	    // Exchange and remove duplicates
	    exchange(non_zeros);
	    sort(non_zeros.begin(), non_zeros.end());
	    unique(non_zeros.begin(), non_zeros.end());
	}
    }

    template <typename Matrix>
    void extract(const Matrix& A, vec_type& v)
    {
	typename traits::row<Matrix>::type             row(A); 
	typename traits::col<Matrix>::type             col(A); 
	typedef typename traits::range_generator<tag::major, Matrix>::type  cursor_type;
	typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
	
	for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor)
	    for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor)
		v.push_back(std::make_pair(row(*icursor), col(*icursor)));
    }

    void local_to_global(vec_type& non_zeros, size_type row_shift, size_type col_shift)
    {
	for (unsigned i= 0; i < non_zeros.size(); i++) {
	    entry_type& nz= non_zeros[i];
	    nz.first+= row_shift; nz.second+= col_shift;
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
	std::vector<vec_type>   send_buffers, recv_buffers;

	for (unsigned i= 0; i < non_zeros.size(); i++) {
	    entry_type& nz= non_zeros[i];
	    if (is_remote(nz)) 
		send_buffers[row_dist.on_rank(nz.first)].push_back(nz);
	}

	all_to_all(communicator(row_dist), send_buffers, recv_buffers);
	
	remove_if(non_zeros.begin(), non_zeros.end(), is_remote);
	for (unsigned p= 0; p < row_dist.size(); p++)
	    if (!recv_buffers[p].empty())
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

