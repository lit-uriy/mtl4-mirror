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



template <typename Matrix>
void test(Matrix& A,  const char* name)
{
    mpi::communicator comm(communicator(A));

    // A= 0.0; // for dense matrices
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

#if 0
    // Serialized output
    wait_for_previous(comm);
    std::cout << "Raw local matrix on proc " << comm.rank() << " is:\n" << A.local_matrix << std::endl;
    start_next(comm);
    std::cout << std::endl;
    comm.barrier();
#endif

    if (!comm.rank()) std::cout << "Matrix is:" << std::endl;
    std::cout << A;
}


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 3) {
	std::cerr << "Example works only for 3 processors!\n";
	env.abort(87);
    }

    std::vector<std::size_t> row_block, col_block;
    row_block.push_back(0); row_block.push_back(4); row_block.push_back(6); row_block.push_back(7); 
    col_block.push_back(0); col_block.push_back(5); col_block.push_back(7); col_block.push_back(7); 

    mtl::par::block_distribution row_dist(row_block), col_dist(col_block);
    matrix::distributed<matrix::compressed2D<double> > A(7, 7, row_dist, col_dist);

    test(A, "compressed2D<double>");
    
    return 0;
}

 














