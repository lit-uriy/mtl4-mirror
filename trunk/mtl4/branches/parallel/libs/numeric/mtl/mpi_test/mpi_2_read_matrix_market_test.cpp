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


#include <boost/mpi.hpp>
#include <iostream>
#include <boost/serialization/string.hpp>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

namespace mpi = boost::mpi;


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);

    std::string program_dir= mtl::io::directory_name(argv[0]);
    std::string fname(argc > 1 ? argv[1] : "matrix_market/laplace_3x4.mtx");
    matrix::distributed<matrix::compressed2D<double> > A(mtl::io::matrix_market(mtl::io::join(program_dir, fname)));
    mtl::par::single_ostream() << "Matrix A is\n " << A << '\n';

    return 0;
}

 














