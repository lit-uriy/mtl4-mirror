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

#include <boost/mpi/status.hpp>
#include <boost/mpi/communicator.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/par/comm_scheme.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl { namespace matrix {


struct dist_mat_cvec_mult_handle 
{
    typedef std::vector<boost::mpi::request> req_type;
    req_type                                 reqs; // send and receive requests
}; 


template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
dist_mat_cvec_mult_handle inline 
dist_mat_cvec_mult_start(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, 
			 tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logical_error("This communication form is not implemented yet"));
    return dist_mat_cvec_mult_handle();
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
dist_mat_cvec_mult_handle inline 
dist_mat_cvec_mult_start(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, 
			 tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    std::cerr << "[nonblocking] initialization" << std::endl;
    typedef typename Matrix::send_structure        send_structure;
    typedef typename Matrix::recv_structure        recv_structure;
    struct dist_mat_cvec_mult_handle handle;

    dist_mat_cvec_mult_fill_send_buffer(A, v);
    typename std::map<int, send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	const send_structure&   s= s_it->second;
	handle.reqs.push_back(communicator(v).isend(s_it->first, 999, &send_buffer(v)[s.offset], size(s.indices))); // pointer and size
    }

    std::cerr << "[nonblocking] Size receive buffer on rank " << communicator(v).rank() << " is " << size(recv_buffer(v)) << std::endl;
    
    typename std::map<int, recv_structure>::const_iterator r_it(A.recv_info.begin()), r_end(A.recv_info.end());

    for (; r_it != r_end; ++r_it) {
	const recv_structure&   s= r_it->second;
	int p= r_it->first;
	handle.reqs.push_back(communicator(v).irecv(p, 999, &recv_buffer(v)[s.offset], s.size)); // pointer and size
    }

    return handle;
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
dist_mat_cvec_mult_handle inline 
dist_mat_cvec_mult_start(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, 
			 tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    return dist_mat_cvec_mult_handle();
}

// Other implementations ....

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
dist_mat_cvec_mult_handle inline dist_mat_cvec_mult_start(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as)
{
    return dist_mat_cvec_mult_start(A, v, w, as, par::comm_scheme(), par::comm_scheme(), par::comm_scheme());
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
boost::mpi::status inline 
dist_mat_cvec_mult_wait(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, dist_mat_cvec_mult_handle& h,
			tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logical_error("This communication form is not implemented yet"));
    return boost::mpi::status();
}

template <typename Matrix, typename Vector>
void inline dist_mat_cvec_mult_fill_send_buffer(const Matrix& A, Vector& v)
{
    typedef typename Collection<Matrix>::size_type size_type;

    MTL_DEBUG_THROW_IF(A.col_dist != distribution(v), incompatible_distribution);
    v.enlarge_send_buffer(A.total_send_size);
    v.enlarge_recv_buffer(A.total_recv_size);

    typename std::map<int, typename Matrix::send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	int p= s_it->first;
	const typename Matrix::send_structure&   s= s_it->second;
	const dense_vector<size_type>&  indices= s.indices;
	for (size_type tgt= s.offset, src= 0; src < size(indices); ++tgt, ++src)
	    send_buffer(v)[tgt]= local(v)[indices[src]];
    }
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
boost::mpi::status inline 
dist_mat_cvec_mult_wait(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, dist_mat_cvec_mult_handle& h,
			tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    typedef typename Collection<Matrix>::size_type size_type;
    // Avoid repeated zeroing of w (= -> +=)
    typedef typename mtl::assign::repeated_assign<Assign>::type assign_mode;
    typedef typename Matrix::recv_structure recv_structure;

    // std::pair<boost::mpi::status,std::vector<boost::mpi::request>::iterator /* TODO: how do I get a generic iterator here, not bound to a vector */> res;
    // Do you mean this with generic iterator? 
    // htor: nope, this is still a vector (what if I want to replace the vector with a list? I have to change this here -> not generic
    // pg: how about this (req_type redefined in dist_mat_cvec_mult_handle)
    std::pair<boost::mpi::status, dist_mat_cvec_mult_handle::req_type::iterator> res;
    
    while(h.reqs.size()) {
      res = boost::mpi::wait_any(h.reqs.begin(), h.reqs.end());
      
      int p=res.first.source();
      if(p == communicator(v).rank()) { // TODO: this is dangerous (not guaranteed by MPI!!!) - talk about other options 
	      // -> How about 2 sets of request and wait only for the receive requests one by one and do waitall on the sends at the (should be finished anyway)
        // htor: this impacts performance significantly ... not good, but I don't know a good alternative (maybe a hash map that translates requests to send or receive reqs.
	// pg: why a hash map? if the requests are stored in a random access iterator we can use a vector<bool> or bit_vector to define is_send_request 
	//     we then just ask if (is_send_request[distance(h.reqs.begin(), res.first)]) ... 
	//     this works also for a list but distance has linear complexity then
        // we have a send request
        h.reqs.erase(res.second);
        std::cerr << "[nonblocking] finished sending my data" << std::endl;
      } else { 
        // we have a receive request 
        h.reqs.erase(res.second);

        const recv_structure s = (*A.recv_info.find(p)).second;

        std::cerr << "[nonblocking] received data from rank " << p << " of size " << s.size << std::endl;
          mat_cvec_mult(const_cast<Matrix&>(A).remote_matrices[p], // Scheiss std::map!!!
		      recv_buffer(v)[irange(s.offset, s.offset + s.size)], local(w), assign_mode());
      }
    }

    boost::mpi::status st;
    return st; // return status of last recv (is there something better?) TODO: bogus, we should return an own status 
    // sounds better but which status, BTW do we need to return at status at all?
    // htor: actually ... I don't think that we need this, if we handle all errors with exceptions
    // pg:   right, we check the status immediately and throw an exception --> I'll do the modifications
}

	
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
boost::mpi::status inline 
dist_mat_cvec_mult_wait(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, dist_mat_cvec_mult_handle& h,
			tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    typedef typename Collection<Matrix>::size_type size_type;
    typedef typename Matrix::send_structure        send_structure;
    typedef typename Matrix::recv_structure        recv_structure;
    // Avoid repeated zeroing of w (= -> +=)
    typedef typename mtl::assign::repeated_assign<Assign>::type assign_mode;

    dist_mat_cvec_mult_fill_send_buffer(A, v);
    typename std::map<int, send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	const send_structure&   s= s_it->second;
	communicator(v).send(s_it->first, 999, &send_buffer(v)[s.offset], size(s.indices)); // pointer and size
    }
    boost::mpi::status st;
    std::cerr << "[blocking] Size receive buffer on rank " << communicator(v).rank() << " is " << size(recv_buffer(v)) << std::endl;
    typename std::map<int, recv_structure>::const_iterator r_it(A.recv_info.begin()), r_end(A.recv_info.end());
    for (; r_it != r_end; ++r_it) {
	const recv_structure&   s= r_it->second;
	int p= r_it->first;
	st= communicator(v).recv(p, 999, &recv_buffer(v)[s.offset], s.size); // pointer and size
	
	mat_cvec_mult(const_cast<Matrix&>(A).remote_matrices[p], // Scheiss std::map!!!
		      recv_buffer(v)[irange(s.offset, s.offset + s.size)], local(w), assign_mode());
    }

    return st; // return status of last recv (is there something better?)
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
boost::mpi::status inline 
dist_mat_cvec_mult_wait(const Matrix& A, VectorIn& v, VectorOut& w, Assign as, dist_mat_cvec_mult_handle& h)
{
    return dist_mat_cvec_mult_wait(A, v, w, as, h, par::comm_scheme(), par::comm_scheme(), par::comm_scheme());
}


template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
void inline dist_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as)
{
    // All three arguments must be distributed
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<Matrix>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorIn>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorOut>::value));

    dist_mat_cvec_mult_handle h(dist_mat_cvec_mult_start(A, v, w, as));
    mat_cvec_mult(local(A), local(v), local(w), as);
    dist_mat_cvec_mult_wait(A, v, w, as, h);
}

}} // namespace mtl::matrix

#endif // MTL_HAS_MPI

#endif // MTL_MATRIX_DIST_MAT_CVEC_MULT_INCLUDE
