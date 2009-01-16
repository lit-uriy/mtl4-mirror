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

namespace mpi = boost::mpi;

inline wait_for_previous(const mpi::communicator& comm)
{
    if (comm.rank() > 0) {
	int xx;
	comm.recv(comm.rank() - 1, 787, xx);
    }
}

inline start_next(const mpi::communicator& comm)
{
    if (comm.rank() < comm.size() - 1)
	comm.send(comm.rank() + 1, 787, 787);
}



template <typename Matrix>
void test(Matrix& A,  const char* name)
{
    mpi::communicator comm(A.communicator());

    A= 0.0; // for dense matrices
    {
	mtl::matrix::inserter<Matrix> ins(A);
	if (comm.rank() == 0) {
	    ins[0][0] << 1.0;
	    ins[1][3] << 3.0;
	    ins[4][1] << 2.0; // remote
	    ins[6][5] << 4.0; // remote
	} else {
	    ins[2][6] << 5.0; // remote
	    ins[3][2] << 6.0; // remote
	    ins[5][4] << 8.0;
	}
    }

    // Serialized output
    wait_for_previous(comm);
    std::cout << "Raw local matrix on proc << " << comm.rank() << " is:\n" << A.raw_local << std::endl;
    start_next(comm);
}


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 2) {
	cerr << "Example works only for 2 processors!\n";
	env.abort(87);
    }

    matrix::distributed<matrix::compressed2D<double> > A(7, 7);

    test(A, "compressed2D<double>");
    
    return 0;
}

 














