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

#ifndef BOOST_MPI_ALL_TO_ALL_PCX_INCLUDE
#define BOOST_MPI_ALL_TO_ALL_PCX_INCLUDE

#ifdef MTL_HAS_MPI

#include <mpi.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/exception.hpp>
#include <boost/mpi/status.hpp>

namespace boost { namespace mpi {

/// Sparse all-to-all communication by Personal Census eXchange.
/** Every processor sends only to a small number of other processors.
    The length of message can very, thus we use vector of vector.
    Based on Hoefler, Siebert, and Lumsdaine: "Scalable Communication Protocols
    for Dynamic Sparse Data Exchange". 
**/
template <typename Vector>
void inline all_to_all_pcx(const communicator& comm, 
			   const std::vector<Vector>& in_values,
			   std::vector<Vector>& out_values)
{
    mtl::par::multiple_ostream<> mout;

    const int my_tag= 99;
    int size = comm.size(), rank = comm.rank();
    
    std::vector<int> is_target(size, 0),     // Whether data is sent to each process 
	             recv_counts(size, 1),   // Array needed by MPI
	             targets;                // collect targets
    int              num_recvs;              // how many messages
   
    for (int dest = 0; dest < size; ++dest) 
	if (!in_values[dest].empty()) {
	    is_target[dest]= 1;
	    targets.push_back(dest);
	}

    BOOST_MPI_CHECK_RESULT(MPI_Reduce_scatter,
			   (&is_target[0], &num_recvs, &recv_counts[0], MPI_INT, MPI_SUM, comm));

    out_values.resize(0); out_values.resize(size); // Clean-up and resize

    // Non-blocking sends
    std::vector<mpi::request> send_req(targets.size());
    for (unsigned i= 0; i < targets.size(); ++i)
	send_req[i]= comm.isend(targets[i], my_tag, in_values[targets[i]]);

    // Reveive messages in random order
    for (int i= 0; i < num_recvs; ++i) {
	status st= comm.probe(any_source, my_tag);
	comm.recv(st.source(), my_tag, out_values[st.source()]);
    }

    wait_all(send_req.begin(), send_req.end());
}

}} // namespace boost::mpi

#endif // MTL_HAS_MPI

#endif // BOOST_MPI_ALL_TO_ALL_PCX_INCLUDE
