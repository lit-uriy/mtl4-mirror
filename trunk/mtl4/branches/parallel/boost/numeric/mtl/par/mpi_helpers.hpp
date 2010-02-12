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

#ifndef MTL_PAR_MPI_HELPERS_INCLUDE
#define MTL_PAR_MPI_HELPERS_INCLUDE

#ifdef MTL_HAS_MPI

#include <string>
#include <boost/mpi/status.hpp>

namespace mtl { namespace par {

/// According message as string for a given error    
std::string inline mpi_error_string(int errorcode)
{
    char arr[MPI_MAX_ERROR_STRING];
    int  len;
    MPI_Error_string(errorcode, arr, &len);
    return std::string(arr, len);
}



}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_MPI_HELPERS_INCLUDE
