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

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>


namespace mtl { namespace vector {

template <typename Vector, typename Distribution>
class distributed
{
public:
    typedef distributed                              self;
    typedef typename Collection<Vector>::size_type   size_type;
    typedef typename Collection<Vector>::value_type  value_type;
    typedef Distribution                             distribution_type;

    typedef Vector                                   local_type;

    /// Constructor for vector with global size \p gsize
    explicit distributed(size_type gsize) : gsize(gsize), dist(gsize), local_vector(dist.num_local(gsize)) {}

    /// Constructor for vector with global size \p gsize and distribution \p dist
    explicit distributed(size_type gsize, const Distribution& dist) 
	: gsize(gsize), dist(dist), local_vector(dist.num_local(gsize))  {}

    friend inline std::ostream& operator<< (std::ostream& out, const self& v) 
    {
	wait_for_previous(v.dist);
	if (v.dist.rank() > 0)
	    out << "||";
	out << v.local_vector;
	out.flush();
	start_next(v.dist);
    }

    template <typename, typename> friend class distributed_inserter;

protected:
    size_type           gsize;
    distribution_type   dist;
    local_type          local_vector;
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


    explicit distributed_inserter(DistributedVector& dist_vector)
	: dist_vector(dist_vector), local_inserter(dist_vector.local_vector),
	  send_buffers(dist_size()), recv_buffers(dist_size())
    {}

    ~distributed_inserter()
    {
	boost::mpi::all_to_all(dist().communicator(), send_buffers, recv_buffers);
	for (unsigned p= 0; p < dist_size(); p++) {
	    const std::vector<entry_type>& my_buffer= recv_buffers[p];
	    for (unsigned i= 0; i < my_buffer.size(); i++)
		update(my_buffer[i].first, my_buffer[i].second);
	}
    }

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
