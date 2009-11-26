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

#include <map>
#include <utility>
#include <vector>
#include <algorithm>


#define MTL_HAS_STD_OUTPUT_OPERATOR // to print std::vector and std::pair
#include <boost/numeric/mtl/mtl.hpp>

#include <parmetis.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

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

    mtl::par::single_ostream sout;
    mtl::par::multiple_ostream<> mout;

    mpi::communicator comm(communicator(A));
    {
	mtl::matrix::inserter<Matrix> mins(A);
	ins<mtl::matrix::inserter<Matrix> > i(mins);
        switch (version) {
          case 1: 
	    switch (comm.rank()) {
	      case 0: i(0, 1); i(0, 2); i(1, 2); i(1, 3); i(2, 3); i(2, 5); std::cout << "version 1\n"; break;
    	      case 1: i(3, 4); i(3, 5); i(4, 5); i(4, 6); break;
    	      case 2: i(5, 6); i(6, 4); i(6, 5);
    	    }; break;
          case 2: 
    	    switch (comm.rank()) {
	      case 0: i(0, 1); i(1, 2); i(2, 3); std::cout << "version 2\n"; break;
    	      case 1: i(3, 4); i(4, 5); break;
    	      case 2: i(5, 6); i(6, 0);
    	  }; break;
        }
    }

    sout << "Matrix is:" << '\n' << A;

    std::vector<idxtype> part;
    int edge_cut= partition_k_way(A, part);

    // mout << "Edge cut = " << edge_cut << ", partition = " << part << '\n';

    mtl::par::block_distribution old_dist(row_distribution(A)), new_dist(old_dist);
    mtl::par::block_migration    migration(old_dist, new_dist);
    parmetis_distribution(old_dist, part, new_dist, migration);

    std::vector<size_type> columns;
    global_columns(A, columns);
    mout << "Global columns = " << columns << '\n';

    std::map<size_type, size_type> new_global;
    new_global_map(migration, columns, new_global);
    mout << "Mapping of columns " << new_global << '\n';
    
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

    matrix::distributed<matrix::compressed2D<double> > A(7, 7), B(7, 7);

    test(A, "compressed2D<double>", 1);
    test(B, "compressed2D<double>", 2);

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












