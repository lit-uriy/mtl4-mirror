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


#include <boost/test/minimal.hpp>
#include <boost/mpi.hpp>
#include <iostream>
#include <cstdlib>
#include <boost/numeric/mtl/mtl.hpp>

namespace mpi = boost::mpi;

template <typename Vector>
void test(Vector& v,  const char* name)
{
    mpi::communicator comm(communicator(v));

    {
	mtl::vector::inserter<Vector> ins(v);
	if (comm.rank() == 0) {
	    ins[0] << 1.0;
	    ins[1] << 3.0;
	    ins[4] << 2.0; // remote
	    ins[6] << 4.0; // remote
	} else {
	    ins[2] << 5.0; // remote
	    ins[3] << 6.0; // remote
	    ins[5] << 8.0;
	}
    }

    if (!comm.rank()) std::cout << "Vector is: ";
    std::cout << v;

    double d= dot(v, v);

    if (comm.rank() == 0) 
	std::cout << "dot(v, v) is: " << d << std::endl;
    if (std::abs(d - 155.0) > 0.001)
	throw "dot(v, v) should be 155!";
}


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 2) {
	std::cerr << "Example works only for 2 processors!\n";
	env.abort(87);
    }

    vector::distributed<dense_vector<double> > v(7);

    test(v, "dense_vector<double>");
    
    return 0;
}

 


