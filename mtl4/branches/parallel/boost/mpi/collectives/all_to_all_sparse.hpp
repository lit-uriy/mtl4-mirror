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

#ifndef BOOST_MPI_ALL_TO_ALL_SPARSE_INCLUDE
#define BOOST_MPI_ALL_TO_ALL_SPARSE_INCLUDE

#ifdef MTL_HAS_MPI

#include <boost/mpi/collectives/all_to_all_pcx.hpp>

namespace boost { namespace mpi {

/// Sparse all-to-all communication
/** Every processor sends only to a small number of other processors.
    The length of message can very, thus we use vector of vector.
    Currently implemented with PCX.
    Later versions might use other methods or dispatch statically
    or dynamically to different algorithms based on
    Hoefler, Siebert, and Lumsdaine: "Scalable Communication Protocols
    for Dynamic Sparse Data Exchange". 
**/
template <typename Vector>
void inline all_to_all_sparse(const communicator& comm, 
			      const std::vector<Vector>& in_values,
			      std::vector<Vector>& out_values)
{
    all_to_all_pcx(comm, in_values, out_values);
}

}} // namespace boost::mpi

#endif

#endif // BOOST_MPI_ALL_TO_ALL_SPARSE_INCLUDE
