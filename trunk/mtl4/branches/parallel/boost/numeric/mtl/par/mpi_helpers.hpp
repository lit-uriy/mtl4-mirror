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
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/par/exception.hpp>

namespace mtl { namespace par {

/// According message as string for a given error    
std::string inline mpi_error_string(int errorcode)
{
    char arr[MPI_MAX_ERROR_STRING];
    int  len;
    MPI_Error_string(errorcode, arr, &len);
    return std::string(arr, len);
}

/// Check MPI error code and throw exception if last operation was not successful
void inline check_mpi(int errorcode)
{
	MTL_THROW_IF(errorcode != MPI_SUCCESS, mpi_error(errorcode));    
}

/// Check MPI error code and throw exception if last operation was not successful
void inline check_mpi(const boost::mpi::status& st)
{
    check_mpi(st.error());
}



}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_MPI_HELPERS_INCLUDE
