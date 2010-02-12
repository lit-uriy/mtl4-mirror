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

#ifndef MTL_MATRIX_DIST_MAT_CVEC_MULT_INCLUDE
#define MTL_MATRIX_DIST_MAT_CVEC_MULT_INCLUDE

#ifdef MTL_HAS_MPI

#include <boost/static_assert.hpp>
#include <boost/mpi/status.hpp>
#include <boost/mpi/communicator.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/par/comm_scheme.hpp>
#include <boost/numeric/mtl/par/mpi_log.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>

namespace mtl { namespace matrix {

    /* Linear operations like y= A * x are performed in the following 5 steps:
       - Scatter x to send buffers x_si
       - Start x_s -> x_r
       - local operation, e.g. local(y)= local(A) * local(x)
       - Finish  x_s -> x_r and wait
       - Reduce local(y) and remote results, e.g. y+= A_i * x_ri
       Transposed operations like w= A^T * v are performed in the following 5 steps:
       - Scatter A^T * v to w_ri (this is no error)
       - Start w_r -> w_s
       - local operation, e.g. local(w)= trans(local(A)) * local(v)
       - Finish w_r -> w_s and wait
       - Reduce local(w) and w_s, e.g. w+= w_si
       The steps of the two kinds of operations are reversed in order and direction.
    */

struct dist_mat_cvec_handle 
{
    typedef std::vector<boost::mpi::request> req_type;
    req_type                                 reqs; // send and receive requests
}; 


template <typename Matrix, typename VectorIn>
dist_mat_cvec_handle inline 
dist_mat_cvec_start(const Matrix& A, const VectorIn& v, 
			 tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logic_error("This communication form is not implemented yet"));
    return dist_mat_cvec_handle();
}


template <typename Matrix, typename VectorIn>
dist_mat_cvec_handle inline 
dist_mat_cvec_start(const Matrix& A, const VectorIn& v, tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    mtl::par::mpi_log << "[nonblocking] initialization" << '\n';
    typedef typename Matrix::send_structure        send_structure;
    typedef typename Matrix::recv_structure        recv_structure;
    struct dist_mat_cvec_handle handle;

    linear_buffer_fill(A, v);
    typename std::map<int, send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	const send_structure&   s= s_it->second;
	handle.reqs.push_back(communicator(v).isend(s_it->first, 999, &send_buffer(v)[s.offset], size(s.indices))); // pointer and size
    }

    mtl::par::mpi_log << "[nonblocking] Size receive buffer on rank " << communicator(v).rank() << " is " << size(recv_buffer(v)) << '\n';
    
    typename std::map<int, recv_structure>::const_iterator r_it(A.recv_info.begin()), r_end(A.recv_info.end());

    for (; r_it != r_end; ++r_it) {
	const recv_structure&   s= r_it->second;
	int p= r_it->first;
	handle.reqs.push_back(communicator(v).irecv(p, 999, &recv_buffer(v)[s.offset], s.size)); // pointer and size
    }

    return handle;
}

template <typename Matrix, typename VectorIn>
dist_mat_cvec_handle inline 
dist_mat_cvec_start(const Matrix& A, const VectorIn& v, tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    return dist_mat_cvec_handle();
}

// Other implementations ....

/// Enlarge send and receive buffers for linear operator application (or its transposed)
template <typename Matrix, typename Vector>
void inline enlarge_buffer(const Matrix& A, Vector& v)
{
    MTL_DEBUG_THROW_IF(*A.cdp != distribution(v), incompatible_distribution());
    v.enlarge_send_buffer(A.total_send_size);
    v.enlarge_recv_buffer(A.total_recv_size);
}


/// Enlarge and fill send buffers for linear operator application like matrix vector product
template <typename Matrix, typename Vector>
void inline linear_buffer_fill(const Matrix& A, Vector& v)
{
    typedef typename Collection<Matrix>::size_type size_type;

    enlarge_buffer(A, v);

    typename std::map<int, typename Matrix::send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	int p= s_it->first;
	const typename Matrix::send_structure&   s= s_it->second;
	const dense_vector<size_type, mtl::vector::parameters<> >&  indices= s.indices; // parameters shouldn't be needed here!
	for (size_type tgt= s.offset, src= 0; src < size(indices); ++tgt, ++src)
	    send_buffer(v)[tgt]= local(v)[indices[src]];
    }
}



template <typename Matrix, typename VectorIn, typename VectorOut, typename Functor>
boost::mpi::status inline 
dist_mat_cvec_wait(const Matrix& A, const VectorIn& v, VectorOut& w, const Functor& op, dist_mat_cvec_handle& h,
			tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logic_error("This communication form is not implemented yet"));
    return boost::mpi::status();
}


template <typename Matrix, typename VectorIn, typename VectorOut, typename Functor>
boost::mpi::status inline 
dist_mat_cvec_wait(const Matrix& A, const VectorIn& v, VectorOut& w, const Functor& op, dist_mat_cvec_handle& h,
			tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    typedef typename Collection<Matrix>::size_type size_type;
    typedef typename Matrix::recv_structure        recv_structure;

    // std::pair<boost::mpi::status,std::vector<boost::mpi::request>::iterator /* TODO: how do I get a generic iterator here, not bound to a vector */> res;
    // Do you mean this with generic iterator? 
    // htor: nope, this is still a vector (what if I want to replace the vector with a list? I have to change this here -> not generic
    // pg: how about this (req_type redefined in dist_mat_cvec_handle)
    std::pair<boost::mpi::status, dist_mat_cvec_handle::req_type::iterator> res;
    
    while(h.reqs.size()) {
	res= boost::mpi::wait_any(h.reqs.begin(), h.reqs.end());
      
	int p= res.first.source();
	if(p == communicator(v).rank()) { // TODO: this is dangerous (not guaranteed by MPI!!!) - talk about other options 
	    // -> How about 2 sets of request and wait only for the receive requests one by one and do waitall on the sends at the (should be finished anyway)
	    // htor: this impacts performance significantly ... not good, but I don't know a good alternative (maybe a hash map that translates requests to send or receive reqs.
	    // pg: why a hash map? if the requests are stored in a random access iterator we can use a vector<bool> or bit_vector to define is_send_request 
	    //     we then just ask if (is_send_request[distance(h.reqs.begin(), res.first)]) ... 
	    //     this works also for a list but distance has linear complexity then
	    // we have a send request
	    h.reqs.erase(res.second);
	    mtl::par::mpi_log << "[nonblocking] finished sending my data" << '\n';
	} else { 
	    // we have a receive request 
	    h.reqs.erase(res.second);

	    const recv_structure s = (*A.recv_info.find(p)).second;

	    mtl::par::mpi_log << "[nonblocking] received data from rank " << p << " of size " << s.size << '\n';
	    op(const_cast<Matrix&>(A).remote_matrices[p], // Scheiss std::map!!!
	       recv_buffer(v)[irange(s.offset, s.offset + s.size)], local(w));
	}
    }

    boost::mpi::status st;
    return st; // return status of last recv (is there something better?) TODO: bogus, we should return an own status 
    // sounds better but which status, BTW do we need to return at status at all?
    // htor: actually ... I don't think that we need this, if we handle all errors with exceptions
    // pg:   right, we check the status immediately and throw an exception --> I'll do the modifications
}

	
template <typename Matrix, typename VectorIn, typename VectorOut, typename Functor>
boost::mpi::status inline 
dist_mat_cvec_wait(const Matrix& A, const VectorIn& v, VectorOut& w, const Functor& op, dist_mat_cvec_handle& h,
			tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    typedef typename Collection<Matrix>::size_type size_type;
    typedef typename Matrix::send_structure        send_structure;
    typedef typename Matrix::recv_structure        recv_structure;

    linear_buffer_fill(A, v);
    typename std::map<int, send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	const send_structure&   s= s_it->second;
	communicator(v).send(s_it->first, 999, &send_buffer(v)[s.offset], size(s.indices)); // pointer and size
    }
    boost::mpi::status st;
    mtl::par::mpi_log << "[blocking] Size receive buffer on rank " << communicator(v).rank() << " is " << size(recv_buffer(v)) << '\n';
    typename std::map<int, recv_structure>::const_iterator r_it(A.recv_info.begin()), r_end(A.recv_info.end());
    for (; r_it != r_end; ++r_it) {
	const recv_structure&   s= r_it->second;
	int p= r_it->first;
	st= communicator(v).recv(p, 999, &recv_buffer(v)[s.offset], s.size); // pointer and size
	
	op(const_cast<Matrix&>(A).remote_matrices[p], // Scheiss std::map!!!
	   recv_buffer(v)[irange(s.offset, s.offset + s.size)], local(w));
    }

    return st; // return status of last recv (is there something better?)
}





template <typename Assign>
struct remote_mat_cvec_mult_functor
{
    // Avoid repeated zeroing of w (= -> +=)
    typedef typename mtl::assign::repeated_assign<Assign>::type assign_mode;

    template <typename Matrix, typename VectorIn, typename VectorOut>
    void operator()(const Matrix& A, const VectorIn& v, VectorOut& w) const
    {
	mat_cvec_mult(A, v, w, assign_mode());
    }
};


// Explicit communication scheme per call
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign,
	  typename Blocking, typename Coll, typename Buffering>
void inline dist_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as, Blocking, Coll, Buffering)
{
    // All three arguments must be distributed
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<Matrix>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorIn>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorOut>::value));

    dist_mat_cvec_handle h(dist_mat_cvec_start(A, v, Blocking(), Coll(), Buffering()));
    mat_cvec_mult(local(A), local(v), local(w), as);
    dist_mat_cvec_wait(A, v, w, remote_mat_cvec_mult_functor<Assign>(), h, Blocking(), Coll(), Buffering());
}


// Use Communication scheme from whole build
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
void inline dist_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as)
{
    dist_mat_cvec_mult(A, v, w, as, par::comm_scheme(), par::comm_scheme(), par::comm_scheme());
}

// Use Communication scheme from whole build
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
void inline trans_dist_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as)
{
    // All three arguments must be distributed
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<Matrix>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorIn>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorOut>::value));

    std::cout << "In trans_dist_mat_cvec_mult\n";
}


}} // namespace mtl::matrix

#endif // MTL_HAS_MPI

#endif // MTL_MATRIX_DIST_MAT_CVEC_MULT_INCLUDE
