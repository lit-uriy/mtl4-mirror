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

template <typename Inserter>
struct ins
{
    typedef typename Inserter::size_type  size_type;
    ins(Inserter& i) : i(i) {}
    void operator()(size_type r, size_type c) {	i[r][c] << 1.0;  }
    Inserter& i;
};

template <typename Matrix>
void test(Matrix& A,  const char* name, int version)
{
    typedef typename mtl::Collection<Matrix>::size_type size_type;
    typedef std::pair<size_type, size_type>             entry_type;
    typedef std::vector<entry_type>                     vec_type;

    mpi::communicator comm(communicator(A));
    {
	mtl::matrix::inserter<Matrix> mins(A);
	ins<mtl::matrix::inserter<Matrix> > i(mins);
        switch (version) {
          case 1: 
	    switch (comm.rank()) {
	      case 0: i(0, 1); i(0, 2); i(1, 2); i(1, 3); i(2, 3); i(2, 5); break;
    	      case 1: i(3, 4); i(3, 5); i(4, 5); i(4, 6); break;
    	      case 2: i(5, 6); i(6, 4); i(6, 5);
    	    }; break;
          case 2: 
    	    switch (comm.rank()) {
    	      case 0: i(0, 1); i(1, 2); i(2, 3); break;
    	      case 1: i(3, 4); i(4, 5); break;
    	      case 2: i(5, 6); i(6, 0);
    	  }; break;
        }
    }

    if (!comm.rank()) std::cout << "Matrix is:" << std::endl;
    std::cout << A; std::cout.flush();

    partition_k_way(A);
}


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

#ifdef MTL_HAS_PARMETIS
    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 3) {
	std::cerr << "Example works only for 3 processors!\n";
	env.abort(87);
    }

    matrix::distributed<matrix::compressed2D<double> > A(7, 7);

    test(A, "compressed2D<double>", 1);
    test(A, "compressed2D<double>", 2);
#else
    std::cout << "Test requires the definition of MTL_HAS_PARMETIS (and of course"
	      << " the presence of ParMetis).\n";
#endif

    return 0;
}

 














