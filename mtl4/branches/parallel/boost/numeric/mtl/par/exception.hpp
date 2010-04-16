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

#include <sstream>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/par/mpi_helpers.hpp>
#include <boost/lexical_cast.hpp>

namespace mtl { namespace par {


/// Exception for MPI errors
struct mpi_error : public runtime_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit mpi_error(const char *s= "MPI error", int ec= 0) : runtime_error(s), ec(ec) {}
    /// Error message according to code
    explicit mpi_error(int ec) : runtime_error("MPI error"), m(message(ec)), ec(ec) {}

    ~mpi_error() throw() {}
    /// Long error message
    virtual const char* what() const throw() { return m.c_str(); }
    /// MPI error code
    int error_code() const { return ec; }

  private:
    std::string message(int ec)
    {
	return std::string("MPI error ") + boost::lexical_cast<std::string>(ec) + ": " + mpi_error_string(ec);
    }

    std::string m;
    int ec;
};


}} // namespace mtl::par

#endif // MTL_PAR_EXCEPTION_INCLUDE
