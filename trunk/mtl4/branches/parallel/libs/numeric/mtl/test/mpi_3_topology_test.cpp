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
#include <boost/test/minimal.hpp>

#if defined(MTL_HAS_PARMETIS) && defined(MTL_HAS_MPI)

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/mpi.hpp>


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;
    typedef matrix::distributed<compressed2D<double> > matrix_type;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    //matrix_type C(mtl::io::matrix_market("matrix.mtx")); // The file is not there!!!
    //matrix_type D(C, parmetis_migration(C));

    return 0;
}

 
#else 

int test_main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_PARMETIS (and of course"
	      << " the presence of ParMetis).\n";
    return 0;
}

#endif












