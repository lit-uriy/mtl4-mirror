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

namespace mtl { namespace par {

/// ostream that writes only on first processor; by default on std::cout using MPI_WORLD
struct single_ostream
{
    /// Constructor for out or std::cout and MPI_WORLD
    single_ostream(std::ostream& out = std::cout) : out(out), comm(boost::mpi::communicator()) {} 
    /// Constructor for out and given communicator
    single_ostream(std::ostream& out, const boost::mpi::communicator& comm) : out(out), comm(comm) {} 
    /// Constructor for out and dist's communicator
    single_ostream(std::ostream& out, const base_distribution& dist) : out(out), comm(communicator(dist)) {} 

    template <typename T> friend single_ostream& operator<<(single_ostream& os, const T& v);

    /// Flush output
    void flush() { if (comm.rank() == 0) out.flush(); }
  private:
    std::ostream&            out;
    boost::mpi::communicator comm;
};


/// The output command
template <typename T>
inline single_ostream& operator<<(single_ostream& os, const T& v)
{
    if(traits::is_distributed<T>::value || os.comm.rank() == 0)
	os.out << v;
    return os;
}


}} // namespace mtl::par

#endif // MTL_HAS_MPI

#endif // MTL_PAR_SINGLE_OSTREAM_INCLUDE
