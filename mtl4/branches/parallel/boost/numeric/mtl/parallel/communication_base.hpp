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

#ifndef MTL_PARALLEL_COMMUNICATION_BASE_INCLUDE
#define MTL_PARALLEL_COMMUNICATION_BASE_INCLUDE

#include <boost/mpi/communicator.hpp>

namespace mtl { namespace parallel {


    template <typename Parameters, bool IsDistributed>
    struct communication_base {
	communication_base(const boost::mpi::communicator&) {}
    };
	


    template <typename Parameters>
    struct communication_base<Parameters, true>
    {
	communication_base(const boost::mpi::communicator& comm) : comm(comm) {}
	
	boost::mpi::communicator comm
    };
   
}} // namespace mtl::parallel

#endif // MTL_PARALLEL_COMMUNICATION_BASE_INCLUDE
