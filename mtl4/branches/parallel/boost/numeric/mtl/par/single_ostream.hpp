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

#ifndef MTL_PAR_SINGLE_OSTREAM_INCLUDE
#define MTL_PAR_SINGLE_OSTREAM_INCLUDE

#ifdef MTL_HAS_MPI

#include <iostream>
#include <string>

#include <boost/mpi/communicator.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>

namespace mtl { namespace par {

/// ostream that writes only on first processor; by default on std::cout using MPI_WORLD
/** Doesn't work with std::endl yet!!! **/
struct single_ostream 
  : public std::ostream 
{
    /// Constructor for out or std::cout and MPI_WORLD
    single_ostream(std::ostream& out = std::cout) : out(out), comm(boost::mpi::communicator()) {} 
    /// Constructor for out and given communicator
    single_ostream(std::ostream& out, const boost::mpi::communicator& comm) : out(out), comm(comm) {} 
    /// Constructor for out and dist's communicator
    single_ostream(std::ostream& out, const base_distribution& dist) : out(out), comm(communicator(dist)) {} 

    template <typename T>
    single_ostream& operator<<(const T& v)
    {
	if(traits::is_distributed<T>::value || comm.rank() == 0)
	    out << v;
	return *this;
    }

    /// Flush output
    void flush() { if (comm.rank() == 0) out.flush(); }
  private:
    std::ostream&            out;
    boost::mpi::communicator comm;
};

inline single_ostream& endl(single_ostream& os)
{
    os << "\n"; os.flush(); return os;
}

}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_SINGLE_OSTREAM_INCLUDE
