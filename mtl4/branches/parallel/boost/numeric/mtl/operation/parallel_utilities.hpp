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

#ifndef MTL_PAR_PARALLEL_UTILITIES_INCLUDE
#define MTL_PAR_PARALLEL_UTILITIES_INCLUDE

#ifdef MTL_HAS_MPI

#include <boost/mpi/communicator.hpp>

namespace boost { namespace mpi {

inline void wait_for_previous(const communicator& comm)
{
    if (comm.rank() > 0) {
	int xx;
	comm.recv(comm.rank() - 1, 787, xx);
    }
}

inline void start_next(const communicator& comm)
{
    if (comm.rank() < comm.size() - 1)
	comm.send(comm.rank() + 1, 787, 787);
}

}} // boost::mpi

namespace mtl { namespace par {

inline void wait_for_previous(const base_distribution& dist)
{
    wait_for_previous(communicator(dist));
}

inline void start_next(const base_distribution& dist)
{
    start_next(communicator(dist));
}



}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_PARALLEL_UTILITIES_INCLUDE
