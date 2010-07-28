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

void test_mpi_log();

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);

    mtl::par::mpi_log << "Alles Kacke.\n";

    test_mpi_log();

    return 0;
}
 

