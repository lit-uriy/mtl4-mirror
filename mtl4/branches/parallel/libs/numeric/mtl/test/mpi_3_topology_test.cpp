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
#include <string>

int test_main(int argc, char* argv[]) 
{
    using namespace mtl;
    typedef matrix::distributed<compressed2D<double> > matrix_type;

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    
    std::string program_dir= io::directory_name(argv[0]), 
                file_name= io::join(program_dir, "matrix_market/topomap_test.mtx");

    //matrix_type C(io::matrix_market("matrix.mtx")); // The file is not there!!!
    //matrix_type C(io::matrix_market(file_name));    // Besser so!!!
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












