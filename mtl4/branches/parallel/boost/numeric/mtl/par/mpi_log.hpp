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

#ifndef MTL_PAR_MPI_LOG_INCLUDE
#define MTL_PAR_MPI_LOG_INCLUDE

#ifdef MTL_HAS_MPI

#include <iostream>

namespace mtl { namespace par {


/// Helper stream type for logging MPI activities, only active if MTL_WITH_MPI_LOG
struct mpi_log_t
{
    template <typename T>
    mpi_log_t& operator<<(const T& v)
    {
#     ifdef MTL_WITH_MPI_LOG
	std::cerr << v;
#     endif
	return *this;
    }
} 
/// Helper stream for logging MPI activities, only active if MTL_WITH_MPI_LOG
mpi_log;

}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_MPI_LOG_INCLUDE
