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

#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <string>

namespace mtl { namespace par {


struct single_ostream
{
    single_ostream(std::ostream& out = std::cout) : out(out), comm(boost::mpi::communicator()) {} 
    single_ostream(std::ostream& out, const boost::mpi::communicator& comm) : out(out), comm(comm) {} 
    single_ostream(std::ostream& out, const base_distribution& dist) : out(out), comm(communicator(dist)) {} 

    template <typename T>
    single_ostream& operator<<(const T& v)
    {
	comm.barrier();
	if (comm.rank() == comm.size()-1) {
	    out << v;
	    out.flush();
	}
	comm.barrier();
	return *this;
    }


    std::ostream&            out;
    boost::mpi::communicator comm;
};



}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_SINGLE_OSTREAM_INCLUDE
