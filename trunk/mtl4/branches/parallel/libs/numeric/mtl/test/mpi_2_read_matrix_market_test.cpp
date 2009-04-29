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
    
    // This is an ugly test to be removed
    if (strlen(argv[0]) > strlen("mpi_2_read_matrix_market_test")+4) {
	std::cerr << "For simplicity this test works only in the test directory\n"
		  << "Please cd there and rerun the test.";
	return 0;
    }

    matrix::distributed<matrix::compressed2D<double> > A(mtl::io::matrix_market("matrix_market/laplace_3x4.mtx"));
    mtl::par::single_ostream() << "Matrix A is\n " << A << '\n';

    // Test not very elegant (and not very complete)
    if (A.row_dist.is_local(7)) {
	int r= A.row_dist.global_to_local(7);
	// if (local(A)[r][6] != -1.0) throw "Should be -1.";
	if (local(A)[r][A.col_dist.global_to_local(7)] != 4.0) throw "Diagonal should be 4.";
	//if (local(A)[r][8] != 0.0) throw "Should be 0.";
    }

    return 0;
}

 














