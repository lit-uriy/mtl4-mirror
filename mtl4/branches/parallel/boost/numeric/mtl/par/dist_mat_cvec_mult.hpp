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
#include <boost/lexical_cast.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/par/comm_scheme.hpp>
#include <boost/numeric/mtl/par/mpi_helpers.hpp>
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
       - Scatter x to send buffers x_si                                     linear_buffer_fill called in dist_mat_cvec_start(nb)/dist_mat_cvec_wait(bl)
       - Start x_s -> x_r                                                   dist_mat_cvec_start(nb)/dist_mat_cvec_wait(bl)
       - local operation, e.g. local(y)= local(A) * local(x)                dist_mat_cvec_op
       - Finish  x_s -> x_r and wait                                        dist_mat_cvec_wait
       - Reduce local(y) and remote results, e.g. y+= A_i * x_ri            dist_mat_cvec_wait
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
dist_mat_cvec_start(const Matrix& A, const VectorIn& v, tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logic_error("This communication form is not implemented yet"));
    return dist_mat_cvec_handle();
}

template <typename Matrix, typename VectorIn>
inline dist_mat_cvec_handle
dist_mat_cvec_send_start(const Matrix& A, const VectorIn& v, tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{
    mtl::par::mpi_log << "[nonblocking] initialization" << '\n';
    typedef typename Matrix::send_structure        send_structure;
    dist_mat_cvec_handle handle;

    typename std::map<int, send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	const send_structure&   s= s_it->second;
	handle.reqs.push_back(communicator(v).isend(s_it->first, 999, &send_buffer(v)[s.offset], size(s.indices))); // pointer and size
    }
    return handle;
}

template <typename Matrix, typename VectorIn>
inline void
dist_mat_cvec_recv_start(const Matrix& A, const VectorIn& v, dist_mat_cvec_handle& handle, tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{
    typedef typename Matrix::recv_structure        recv_structure;

    mtl::par::mpi_log << "[nonblocking] Size receive buffer on rank " << communicator(v).rank() << " is " << size(recv_buffer(v)) << '\n';    
    typename std::map<int, recv_structure>::const_iterator r_it(A.recv_info.begin()), r_end(A.recv_info.end());
    for (; r_it != r_end; ++r_it) {
	const recv_structure&   s= r_it->second;
	int p= r_it->first;
	handle.reqs.push_back(communicator(v).irecv(p, 999, &recv_buffer(v)[s.offset], s.size)); // pointer and size
    }
}

template <typename Matrix, typename VectorIn>
dist_mat_cvec_handle inline 
dist_mat_cvec_start(const Matrix& A, const VectorIn& v, tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    linear_buffer_fill(A, v);
    dist_mat_cvec_handle handle= dist_mat_cvec_send_start(A, v, tag::comm_non_blocking(), tag::comm_p2p(), tag::comm_buffer());
    dist_mat_cvec_recv_start(A, v, handle, tag::comm_non_blocking(), tag::comm_p2p(), tag::comm_buffer());
    return handle;
}

// Do nothing for all blocking schemes
template <typename Matrix, typename VectorIn>
dist_mat_cvec_handle inline 
dist_mat_cvec_start(const Matrix& A, const VectorIn& v, tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
//dist_mat_cvec_start(const Matrix& A, const VectorIn& v, tag::comm_blocking, tag::universe, tag::universe) // doesn't work
{ 
    return dist_mat_cvec_handle();
}

// Other implementations ....

/// Enlarge send and receive buffers for linear operator application (or its transposed)
template <typename Matrix, typename Vector>
void inline enlarge_buffer(const Matrix& A, const Vector& v)
{    
    // std::cerr << "col(A) = " << col_distribution(A) << ", dist(v) = " << distribution(v) << std::endl;
    MTL_DEBUG_THROW_IF(col_distribution(A) != distribution(v), incompatible_distribution());
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
	//int p= s_it->first;
	const typename Matrix::send_structure&   s= s_it->second;
	const dense_vector<size_type, mtl::vector::parameters<> >&  indices= s.indices; // parameters shouldn't be needed here!
	for (size_type tgt= s.offset, src= 0; src < size(indices); ++tgt, ++src)
	    send_buffer(v)[tgt]= local(v)[indices[src]];
    }
}


template <typename Matrix, typename VectorIn, typename VectorOut, typename Functor>
void inline 
dist_mat_cvec_wait(const Matrix& A, const VectorIn& v, VectorOut& w, const Functor& op, dist_mat_cvec_handle& h,
			tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logic_error("This communication form is not implemented yet"));
}


template <typename Matrix, typename VectorIn, typename VectorOut, typename Functor>
void inline 
dist_mat_cvec_wait(const Matrix& A, const VectorIn& v, VectorOut& w, const Functor& op, dist_mat_cvec_handle& h,
			tag::comm_non_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    typedef typename Collection<Matrix>::size_type size_type;
    typedef typename Matrix::recv_structure        recv_structure;

    while(h.reqs.size()) { // see (1) at file end
	std::pair<boost::mpi::status, dist_mat_cvec_handle::req_type::iterator> res= boost::mpi::wait_any(h.reqs.begin(), h.reqs.end());
	// mtl::par::check_mpi(res.first); // error code is non-sense !!!, besides boost::mpi checks already

	int p= res.first.source();
	if(p == communicator(v).rank()) 
	    mtl::par::mpi_log << "[nonblocking] finished sending my data" << '\n';
	else { 
	    // we have a receive request 
	    const recv_structure& s = (*A.recv_info.find(p)).second;
	    mtl::par::mpi_log << "[nonblocking] received data from rank " << p << " of size " << s.size << '\n';
	    op(const_cast<Matrix&>(A).remote_matrices[p] /* Scheiss std::map */, recv_buffer(v)[irange(s.offset, s.offset + s.size)], local(w));
	}
	h.reqs.erase(res.second);
    }
}

	
template <typename Matrix, typename VectorIn, typename VectorOut, typename Functor>
void inline 
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
	communicator(v).send(s_it->first, 999, &send_buffer(v)[s.offset], size(s.indices)); 
    }

    mtl::par::mpi_log << "[blocking] Size receive buffer on rank " << communicator(v).rank() << " is " << size(recv_buffer(v)) << '\n';
    typename std::map<int, recv_structure>::const_iterator r_it(A.recv_info.begin()), r_end(A.recv_info.end());
    for (; r_it != r_end; ++r_it) {
	const recv_structure&   s= r_it->second;
	int                     p= r_it->first;
	mtl::par::check_mpi( communicator(v).recv(p, 999, &recv_buffer(v)[s.offset], s.size) );
	op(const_cast<Matrix&>(A).remote_matrices[p] /* Scheiss std::map */, recv_buffer(v)[irange(s.offset, s.offset + s.size)], local(w));
    }
}


template <typename Assign>
struct mat_cvec_mult_functor
{
    template <typename Matrix, typename VectorIn, typename VectorOut>
    void operator()(const Matrix& A, const VectorIn& v, VectorOut& w) const
    {
	mat_cvec_mult(A, v, w, Assign());
    }
};

template <typename Assign>
struct trans_mat_cvec_mult_functor
{
    template <typename Matrix, typename VectorIn, typename VectorOut>
    void operator()(const Matrix& A, const VectorIn& v, VectorOut& w) const
    {
	mat_cvec_mult(trans(A), v, w, Assign());
    }
};


// Explicit communication scheme per call
template <typename Matrix, typename VectorIn, typename VectorOut, typename Local, typename Remote,
	  typename Blocking, typename Coll, typename Buffering>
void inline dist_mat_cvec_op(const Matrix& A, const VectorIn& v, VectorOut& w, const Local& local_op, const Remote& remote_op,
			     Blocking, Coll, Buffering)
{
    // All three arguments must be distributed
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<Matrix>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorIn>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorOut>::value));

    dist_mat_cvec_handle h(dist_mat_cvec_start(A, v, Blocking(), Coll(), Buffering()));
    local_op(local(A), local(v), local(w));
    dist_mat_cvec_wait(A, v, w, remote_op, h, Blocking(), Coll(), Buffering());
}


// Use Communication scheme from whole build
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
void inline dist_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as)
{
    // Avoid repeated zeroing of w with repeated matrix vector product (= -> +=)
    typedef typename mtl::assign::repeated_assign<Assign>::type remote_assign;

    typedef mat_cvec_mult_functor<Assign>        local_functor;
    typedef mat_cvec_mult_functor<remote_assign> remote_functor;

    dist_mat_cvec_op(A, v, w, local_functor(), remote_functor(),
		     par::comm_scheme(), par::comm_scheme(), par::comm_scheme());
}

template <typename Matrix, typename VectorIn, typename VectorOut>
dist_mat_cvec_handle inline 
trans_dist_mat_cvec_start(const Matrix& A, const VectorIn& v, VectorOut& w, tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logic_error("This communication form is not implemented yet"));
    return dist_mat_cvec_handle();
}

template <typename Matrix, typename VectorOut>
dist_mat_cvec_handle inline 
trans_dist_mat_cvec_start(const Matrix& A, VectorOut& w, tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logic_error("This communication form is not implemented yet"));
    return dist_mat_cvec_handle();
}

// Do nothing for all blocking schemes
template <typename Matrix, typename VectorOut>
dist_mat_cvec_handle inline 
trans_dist_mat_cvec_start(const Matrix& A, VectorOut& w, tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    return dist_mat_cvec_handle();
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Functor>
void inline trans_compute_send_buffer(const Matrix& A, const VectorIn& v, VectorOut& w, Functor op)
{
    enlarge_buffer(A, w);
    typename Matrix::remote_map_const_iterator A_it= A.remote_matrices.begin(), A_end= A.remote_matrices.end();
    for (; A_it != A_end; ++A_it) {
	const typename Matrix::recv_structure& r= A.recv_info.find(A_it->first)->second;
	typename DistributedCollection<VectorOut>::local_type w_sub(recv_buffer(w)[irange(r.offset, r.offset + r.size)]); // might need to handle sub-matrix with type trait 
	// std::cout << "trans_compute_send_buffer on " << A.row_dist.rank() << " recv_buffer is " << recv_buffer(w) << ", w_sub is " << w_sub << std::endl;
	op(A_it->second, local(v), w_sub);
	// std::cout << "trans_compute_send_buffer on " << A.row_dist.rank() << " for proc " << A_it->first << ": A is \n"
	// 	  << A_it->second << "local(v) is " << local(v) << ", trans(A)*v is " << recv_buffer(w)[irange(r.offset, r.offset + r.size)] << std::endl;
    }
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
void inline 
trans_dist_mat_cvec_wait(const Matrix&, const VectorIn&, VectorOut&, Assign, dist_mat_cvec_handle&,
			tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logic_error("This communication form is not implemented yet"));
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
void inline 
trans_dist_mat_cvec_wait(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as, dist_mat_cvec_handle& h,
			 tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    typedef typename Collection<Matrix>::size_type size_type;
    typedef typename Matrix::send_structure        send_structure;
    typedef typename Matrix::recv_structure        recv_structure;

    // Roles of send and receive buffers are interchanged in transposed operations
    typename std::map<int, recv_structure>::const_iterator r_it(A.recv_info.begin()), r_end(A.recv_info.end());
    for (; r_it != r_end; ++r_it) {
	const recv_structure&   s= r_it->second;
	//int                     p= r_it->first;
	communicator(v).send(r_it->first, 999, &recv_buffer(w)[s.offset], s.size);
    }

    typename std::map<int, send_structure>::const_iterator s_it(A.send_info.begin()), s_end(A.send_info.end());
    for (; s_it != s_end; ++s_it) {
	const send_structure&   s= s_it->second;
	mtl::par::check_mpi( communicator(w).recv(s_it->first, 999, &send_buffer(w)[s.offset], size(s.indices)));

	const dense_vector<size_type, mtl::vector::parameters<> >&  indices= s.indices; // parameters shouldn't be needed here!
	for (size_type tgt= s.offset, src= 0; src < size(indices); ++tgt, ++src)
	    as.update(local(w)[indices[src]], send_buffer(w)[tgt]);
    }
}


// A is matrix itself not transposed_view< ... >
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign, 
	  typename Blocking, typename Coll, typename Buffering>
void inline trans_dist_mat_cvec_op(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as, Blocking, Coll, Buffering)
{
    // All three arguments must be distributed
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<Matrix>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorIn>::value));
    BOOST_STATIC_ASSERT((mtl::traits::is_distributed<VectorOut>::value));
    
    trans_mat_cvec_mult_functor<Assign>                         local_op;
    trans_mat_cvec_mult_functor<assign::assign_sum>             remote_op; // assign because buffer must be initialized
    typename mtl::assign::repeated_assign<Assign>::type         assign;    // how to incorporate remote results

    trans_compute_send_buffer(A, v, w, remote_op);
    dist_mat_cvec_handle h(trans_dist_mat_cvec_start(A, w, Blocking(), Coll(), Buffering()));
    local_op(local(A), local(v), local(w));
    trans_dist_mat_cvec_wait(A, v, w, assign, h, Blocking(), Coll(), Buffering());
}


// Use Communication scheme from whole build, A is transposed_view< ... >
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
void inline trans_dist_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign as)
{
    trans_dist_mat_cvec_op(A.ref, v, w, as, par::comm_scheme(), par::comm_scheme(), par::comm_scheme());
}


}} // namespace mtl::matrix

#endif // MTL_HAS_MPI

#endif // MTL_MATRIX_DIST_MAT_CVEC_MULT_INCLUDE




// (1): this is dangerous (not guaranteed by MPI!!!) - talk about other options 
// -> How about 2 sets of request and wait only for the receive requests one by one and do waitall on the sends at the (should be finished anyway)
// htor: this impacts performance significantly ... not good, but I don't know a good alternative (maybe a hash map that translates requests to send or receive reqs.
// pg: why a hash map? if the requests are stored in a random access iterator we can use a vector<bool> or bit_vector to define is_send_request 
//     we then just ask if (is_send_request[distance(h.reqs.begin(), res.first)]) ... 
//     this works also for a list but distance has linear complexity then
// we have a send request
