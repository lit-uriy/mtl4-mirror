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

    
    {
    mpi::environment env(argc, argv);
#if 0 // What's wrong with that????
    std::string fname= mtl::io::join(mtl::io::directory_name(argv[0]), "matrix_market/laplace_3x4.mtx");
    matrix::distributed<matrix::compressed2D<double> > A(mtl::io::matrix_market(fname));
#endif

    std::string program_dir= mtl::io::directory_name(argv[0]);
    matrix::distributed<matrix::compressed2D<double> > A(mtl::io::matrix_market(mtl::io::join(program_dir, argv[1])));
    //mtl::par::single_ostream() << "Matrix A is\n " << A << '\n';

    // -- number of neighbors -- A.send_info.size();
    mpi::communicator world;
    int v=A.send_info.size();
    std::vector<int> all;
    gather(world, v, all, 0);
    //std::cout << "["<<world.rank()<<"] "<<A.send_info.size()<<"\n";
    if(!world.rank()) {
      for(int i=0; i<world.size(); i++) std::cout << all[i] << "\n";
    }
  

    // Test not very elegant (and not very complete)
    /*if (row_distribution(A).is_local(7)) {
	int r= A.row_dist.global_to_local(7);
	// if (local(A)[r][6] != -1.0) throw "Should be -1.";
	if (local(A)[r][A.cdp->global_to_local(7)] != 4.0) throw "Diagonal should be 4.";
	//if (local(A)[r][8] != 0.0) throw "Should be 0.";
    }*/
    }

    return 0;
}

 














