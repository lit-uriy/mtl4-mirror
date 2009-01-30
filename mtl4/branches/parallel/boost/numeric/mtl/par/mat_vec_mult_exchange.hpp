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

#ifndef MTL_VECTOR_MAT_VEC_MULT_EXCHANGE_INCLUDE
#define MTL_VECTOR_MAT_VEC_MULT_EXCHANGE_INCLUDE

#ifdef MTL_HAS_MPI

#include <boost/mpi/status.hpp>
#include <boost/mpi/communicator.hpp>

#include <boost/numeric/mtl/par/comm_scheme.hpp>

namespace mtl { namespace vector {


struct mat_vec_mult_handle {}; // dummy (so far)


template <typename Matrix, typename Vector>
mat_vec_mult_handle inline mat_vec_mult_start(const Matrix& A, Vector& v, 
					      tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logical_error("Communication combination not implemented yet"));
    return mat_vec_mult_handle();
}

template <typename Matrix, typename Vector>
mat_vec_mult_handle inline mat_vec_mult_start(const Matrix& A, Vector& v, 
					      tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    return mat_vec_mult_handle();
}

// Other implementations ....

template <typename Matrix, typename Vector>
mat_vec_mult_handle inline mat_vec_mult_start(const Matrix& A, Vector& v)
{
    return mat_vec_mult_start(A, v, par::comm_scheme(), par::comm_scheme(), par::comm_scheme());
}

template <typename Matrix, typename Vector>
boost::mpi::status inline mat_vec_mult_wait(const Matrix& A, Vector& v, const mat_vec_mult_handle& h,
					    tag::universe, tag::universe, tag::universe)
{ 
    MTL_THROW(logical_error("Communication combination not implemented yet"));
    return boost::mpi::status();
}
	
template <typename Matrix, typename Vector>
boost::mpi::status inline mat_vec_mult_wait(const Matrix& A, Vector& v, const mat_vec_mult_handle& h,
					    tag::comm_blocking, tag::comm_p2p, tag::comm_buffer)
{ 
    typedef Collection<Matrix>::size_type size_type;

    MTL_DEBUG_THROW_IF(A.col_dist != distribution(v), incompatible_distribution);
    v.expand_send_buffer(A.total_send_size);
    v.expand_recv_buffer(A.total_recv_size);

    typename std::map<int, send_structure>::const_iterator sit(A.send_info.begin()), send(A.send_info.end());
    for (; sit != send; ++sit) {
	int p= sit->first;
	const Matrix::send_structure& s= sit->second;
	for (size_type i= s.offset, 
	size_type offset(s.offset), 
    }

    typename std::map<int, recv_structure>::const_iterator rit(A.recv_info.begin()), rend(A.recv_info.end());
    for (; rit != rend; ++rit) {
	
    }




    return boost::mpi::status();
}

template <typename Matrix, typename Vector>
boost::mpi::status inline mat_vec_mult_wait(const Matrix& A, Vector& v, const mat_vec_mult_handle& h)
{
    return mat_vec_mult_wait(A, v, par::comm_scheme(), par::comm_scheme(), par::comm_scheme());
}


template <typename Matrix, typename Vector>
mat_vec_mult_handle inline mat_vec_mult(const Matrix& A, Vector& v)
{
    mat_vec_mult_handle h= mat_vec_mult_start(A, v);
    return mat_vec_mult_wait(A, v, h);
}

}} // namespace mtl::vector

#endif // MTL_VECTOR_MAT_VEC_MULT_EXCHANGE_INCLUDE
