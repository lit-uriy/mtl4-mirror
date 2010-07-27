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
#include <utility>
#include <vector>
#include <algorithm>

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/test/minimal.hpp>

#define MTL_HAS_STD_OUTPUT_OPERATOR // to print std::vector and std::pair
#include <boost/numeric/mtl/mtl.hpp>

namespace mpi = boost::mpi;

template <typename Vector>
struct ins
{
    typedef typename Vector::value_type      entry_type;
    typedef typename entry_type::first_type  size_type;

    ins(Vector& v) : v(v) {}
    void operator()(size_type r, size_type c)
    {
	v.push_back(entry_type(r, c));
    }

    Vector& v;
};

template <typename Matrix>
void test(Matrix& A,  const char* name)
{
    typedef typename mtl::Collection<Matrix>::size_type size_type;
    typedef std::pair<size_type, size_type>             entry_type;
    typedef std::vector<entry_type>                     vec_type;

    mpi::communicator comm(communicator(A));

    // A= 0.0; // for dense matrices
    {
	mtl::matrix::inserter<Matrix> ins(A);
	if (comm.rank() == 0) {
	    ins[0][0] << 1.0;
	    ins[1][3] << 3.0;
	    ins[2][2] << 9.0;
	    ins[3][1] << 9.0; // a[j][i] is present
	    ins[2][5] << 7.0;
	    ins[4][1] << 2.0; // remote
	    ins[6][5] << 4.0; // remote
	} else {
	    ins[2][6] << 5.0; // remote
	    ins[3][2] << 6.0; // remote
	    ins[5][4] << 8.0;
	    ins[6][4] << 9.0;
	}
    }

    if (!comm.rank()) std::cout << "Matrix is:" << std::endl;
    std::cout << A; std::cout.flush();

    vec_type non_zeros;
    global_non_zeros(A, non_zeros, true, false);
    mtl::par::multiple_ostream<> mout;
    mout << "Symmetric non-zero entries are " << non_zeros << '\n'; mout.flush();

    vec_type         cmp;
    ins<vec_type>    i(cmp);
    switch (comm.rank()) {
      case 0: i(1, 3); i(1, 4); i(2, 3); i(2, 5); i(2, 6);
  	      i(3, 1); i(3, 2); i(4, 1); i(4, 5); i(4, 6); break;
      case 1: i(5, 2); i(5, 4); i(5, 6); i(6, 2); i(6, 4); i(6, 5); break;
      case 2: break; // leave empty
      default: throw "Only defined for 3 processes.";
    }

    if (non_zeros != cmp) {
	std::cerr << "On rank " << comm.rank() << " wrong non-zeros are calculated!\n"
		  << "Should be " << cmp << "\nGot " << non_zeros << std::endl;
	throw "Wrong result";
    }
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
    row_block.push_back(0); row_block.push_back(5); row_block.push_back(7); row_block.push_back(7); 
    col_block.push_back(0); col_block.push_back(4); col_block.push_back(6); col_block.push_back(7); 

    mtl::par::block_distribution row_dist(row_block), col_dist(col_block);
    matrix::distributed<matrix::compressed2D<double> > A(7, 7, row_dist, col_dist);

    test(A, "compressed2D<double>");
    
    return 0;
}

 














