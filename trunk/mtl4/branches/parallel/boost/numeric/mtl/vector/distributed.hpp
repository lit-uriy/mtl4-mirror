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

#ifndef MTL_VECTOR_DISTRIBUTED_INCLUDE
#define MTL_VECTOR_DISTRIBUTED_INCLUDE

#ifdef MTL_HAS_MPI

#include <iostream>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/all_to_all.hpp>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/vector/crtp_base_vector.hpp>


namespace mtl { namespace vector {

namespace mpi = boost::mpi;

template <typename Vector, typename Distribution>
class distributed
  : public vec_expr< distributed<Vector, Distribution> >,
    public crtp_base_vector< distributed<Vector, Distribution>,
			     typename Collection<Vector>::value_type,
			     typename Collection<Vector>::size_type>
{
public:
    typedef distributed                              self;
    typedef typename Collection<Vector>::size_type   size_type;
    typedef typename Collection<Vector>::value_type  value_type;
    typedef Distribution                             distribution_type;
    typedef crtp_vector_assign< self, value_type, size_type >          assign_base;

    typedef Vector                                   local_type;
    typedef Vector                                   remote_type; // Do we need this (except for DistributedCollection?)
    typedef dense_vector<value_type>                 buffer_type;

    /// Constructor for vector with global size \p gsize
    explicit distributed(size_type gsize) : gsize(gsize), dist(gsize), local_vector(dist.num_local(gsize)) {}

    /// Constructor for vector with global size \p gsize and distribution \p dist
    explicit distributed(size_type gsize, const Distribution& dist) 
	: gsize(gsize), dist(dist), local_vector(dist.num_local(gsize))  {}

    /// Constructor for vector with global size \p gsize and distribution \p dist
    /** Uses default distribution **/
    explicit distributed(size_type gsize, value_type value) 
      : gsize(gsize), dist(gsize), local_vector(dist.num_local(gsize), value) 
    {}
    

    self& operator=(self src)
    {
	assert(this != &src);
	MTL_DEBUG_THROW_IF( gsize != src.gsize, incompatible_size());
	MTL_DEBUG_THROW_IF( dist != src.dist, incompatible_distribution());
	// If variable assigned then src is already a copy, no need to deep copy members
	swap(local_vector, src.local_vector);
	return *this;
    }	

    using assign_base::operator=;

    value_type& operator() (size_type n) { return local_vector(dist.global_to_local(n)); }
    const value_type& operator() (size_type n) const { return local_vector(dist.global_to_local(n)); }

    value_type& operator[] (size_type n) { return local_vector[dist.global_to_local(n)]; }
    const value_type& operator[] (size_type n) const { return local_vector[dist.global_to_local(n)]; }

    void check_dim( size_type s ) const
    {
	MTL_DEBUG_THROW_IF( gsize != 0 && gsize != s, incompatible_size());
    }

    void change_dim(size_type n) { local_vector.change_dim(dist.num_local(n)); }
    
    friend inline std::ostream& operator<< (std::ostream& out, const self& v) 
    {
	std::vector<local_type> all_vectors;
	gather(communicator(v), local(v), all_vectors, 0);
	if (communicator(v).rank() == 0)
	    for (unsigned i= 0; i < all_vectors.size(); i++) {
		if (i) out << "||";
		out << all_vectors[i];
	    }
	return out;
    }

    friend inline const distribution_type& distribution(const self& d) { return d.dist; }
    friend inline const boost::mpi::communicator& communicator(const self& d) { return communicator(d.dist); }
			  
    template <typename, typename> friend class distributed_inserter;

    friend inline size_type num_rows(const self& v) { return mtl::traits::is_row_major<self>::value ? 1 : v.gsize; }
    friend inline size_type num_cols(const self& v) { return mtl::traits::is_row_major<self>::value ? v.gsize : 1; }
    friend inline size_type size(const self& v) { return v.gsize; }

    // Also as member functions because it is used in MTL4 all over the place (needs refactoring some day)
    size_type size() const { return gsize; }

    void delay_assign() const {}

    // Enlarge send buffer so that at least n entries can be sent
    void enlarge_send_buffer(size_type n) const { send_buffer.change_dim(std::max(send_buffer.size(), n)); }
    // Enlarge receive buffer so that at least n entries can be received
    void enlarge_recv_buffer(size_type n) const { recv_buffer.change_dim(std::max(recv_buffer.size(), n)); }

    void release_send_buffer(size_type n) const { send_buffer.change_dim(0); }
    void release_recv_buffer(size_type n) const { recv_buffer.change_dim(0); }

    void set_send_buffer(size_type i, value_type v) { send_buffer[i]= v; }
    value_type get_recv_buffer(size_type i) { return recv_buffer[i]; }

    friend inline local_type& local(self& d) { return d.local_vector; }
    friend inline const local_type& local(const self& d) { return d.local_vector; }

    friend inline buffer_type& send_buffer(const self& d) { return d.send_buffer; }
    friend inline buffer_type& recv_buffer(const self& d) { return d.recv_buffer; }
    // friend inline const buffer_type& recv_buffer(const self& d) { return d.recv_buffer; }

protected:
    size_type           gsize;
    distribution_type   dist;
    local_type          local_vector;
    mutable dense_vector<value_type> send_buffer, recv_buffer;
};

template <typename DistributedVector, 
	  typename Updater = mtl::operations::update_store<typename Collection<DistributedVector>::value_type> >
class distributed_inserter
{
    typename DistributedVector::distribution_type const& dist() const { return dist_vector.dist; }
    int rank() const { return dist().rank(); }
    int dist_size() const { return dist().size(); }

public:
    typedef distributed_inserter                                  self;
    typedef DistributedVector                                     dist_vector_type;
    typedef typename DistributedVector::local_type                local_vector_type;
    typedef typename Collection<DistributedVector>::size_type     size_type;
    typedef typename Collection<DistributedVector>::value_type    value_type;
    typedef std::pair< size_type, value_type >                    entry_type;
    typedef update_proxy<self, size_type>                         proxy_type;


    explicit distributed_inserter(DistributedVector& dist_vector)
	: dist_vector(dist_vector), local_inserter(dist_vector.local_vector),
	  send_buffers(dist_size()), recv_buffers(dist_size())
    {}

    ~distributed_inserter()
    {
	boost::mpi::all_to_all(communicator(dist()), send_buffers, recv_buffers);
	for (unsigned p= 0; p < dist_size(); p++) {
	    const std::vector<entry_type>& my_buffer= recv_buffers[p];
	    for (unsigned i= 0; i < my_buffer.size(); i++)
		update(my_buffer[i].first, my_buffer[i].second);
	}
    }

    proxy_type operator() (size_type n) { return proxy_type(*this, n); }
    proxy_type operator[] (size_type n) { return proxy_type(*this, n); }

    template <typename Modifier>
    void modify(size_type n, value_type value)
    {
	typename DistributedVector::distribution_type const& dist= dist_vector.dist;
	if (dist.is_local(n)) 
	    local_inserter.modify<Modifier>(dist.global_to_local(n), value);
	else
	    send_buffers[dist.on_rank(n)].push_back(std::make_pair(n, value));
    }

    void update(size_type n, value_type value) { modify<Updater>(n, value); }

private:
    DistributedVector&                     dist_vector;
    inserter<local_vector_type, Updater>   local_inserter;
    std::vector<std::vector<entry_type> >  send_buffers, recv_buffers;
};

}} // namespace mtl::vector

#endif // MTL_HAS_MPI

#endif // MTL_VECTOR_DISTRIBUTED_INCLUDE
