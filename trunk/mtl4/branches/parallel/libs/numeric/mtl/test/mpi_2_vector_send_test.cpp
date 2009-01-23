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
#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

namespace mpi = boost::mpi;


int test_main(int argc, char* argv[]) 
{
    mpi::environment env(argc, argv);
    mpi::communicator world;

    if (world.size() != 2) {
	std::cerr << "Example works only for 2 processors!\n";
	env.abort(87);
    }

    mtl::dense_vector<int> v(world.rank() ? 0 : 2);

    if (world.rank() == 0) {
	v[0]= 4 - world.rank(), v[1]= 9;
	world.send(1, 0, v);
    } else {
	std::cout << "v on proc 1 is " << v << std::endl;
	world.recv(0, 0, v);
	std::cout << "v on proc 1 is " << v << std::endl;
	if (v[0] != 4) throw "v[0] should be 4!";
    }

    return 0;
}

 














