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

#include <iostream>
#include <complex>
#include <cmath>
// #include <boost/test/minimal.hpp>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>



int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);

    //std::cout  << "Hallo" << " I am rank " << boost::mpi::communicator().rank() << "\n";

    mtl::par::nosync_rank_ostream nrout;
    nrout << "Hallo" << " I am rank " << boost::mpi::communicator().rank() << " and print without synchronization\n";
    std::cout.flush();

    mtl::par::rank_ostream rout;
    rout << "Hallo" << " I am rank " << boost::mpi::communicator().rank() << " and print with synchronization\n";
    rout << "Hallo again" << " I am rank " << boost::mpi::communicator().rank() << " and print with synchronization\n";

    return 0;
}
 

