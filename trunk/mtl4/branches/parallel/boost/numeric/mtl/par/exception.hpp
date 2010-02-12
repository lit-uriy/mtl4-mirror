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

#ifndef MTL_PAR_EXCEPTION_INCLUDE
#define MTL_PAR_EXCEPTION_INCLUDE

#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/par/mpi_helpers.hpp>
#include <boost/lexical_cast.hpp>

namespace mtl { namespace par {


/// Exception for MPI errors
struct mpi_error : public runtime_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit mpi_error(const char *s= "MPI error", int error_code= 0) : runtime_error(s), error_code(error_code) {}
    explicit mpi_error(int error_code) : runtime_error(message(error_code)), error_code(error_code) {}

    ~mpi_error() throw() {}

    const char* message(int error_code)
    {
	m= "MPI error " + boost::lexical_cast<std::string>(error_code) + ": " + mpi_error_string(error_code);
	return m.c_str();
    }

    std::string m;
    int error_code;
};


}} // namespace mtl::par

#endif // MTL_PAR_EXCEPTION_INCLUDE
