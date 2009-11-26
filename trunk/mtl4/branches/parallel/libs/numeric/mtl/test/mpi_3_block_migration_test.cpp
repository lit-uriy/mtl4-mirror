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

#include <utility>
#include <vector>
#include <algorithm>


#define MTL_HAS_STD_OUTPUT_OPERATOR // to print std::vector and std::pair
#include <boost/numeric/mtl/mtl.hpp>

#include <parmetis.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

namespace mpi = boost::mpi;
using namespace std;


// Check if local old index is correctly projected to new global index
void inline cn(size_t old_local, size_t new_global, const mtl::par::block_migration& migration)
{
    if (migration.new_global(old_local) != new_global)
	throw "Wrong new global index!\n";
}

// Check if local new index is correctly projected to old global index
void inline co(size_t new_local, size_t old_global, const mtl::par::block_migration& migration)
{
    if (migration.old_global(new_local) != old_global)
	throw "Wrong old global index!\n";
}

void test()
{
    mtl::par::single_ostream sout;
    mtl::par::multiple_ostream<> mout;

    mtl::par::parmetis_index_vector part;
    mpi::communicator comm;
    switch (comm.rank()) {
      case 0: part.push_back(1); part.push_back(0); part.push_back(1); break;
      case 1: part.push_back(0); part.push_back(0); break;
      case 2: part.push_back(2); part.push_back(2); break;
    }

    mtl::par::block_distribution old_dist(7), new_dist(old_dist);
    mtl::par::block_migration    migration(old_dist, new_dist);
    parmetis_distribution(old_dist, part, new_dist, migration);

    switch (comm.rank()) {
      case 0: cn(0, 3, migration); cn(1, 0, migration); cn(2, 4, migration); break;
      case 1: cn(0, 1, migration); cn(1, 2, migration); break;
      case 2: cn(0, 5, migration); cn(1, 6, migration); break;
    }
    
    switch (comm.rank()) {
      case 0: co(0, 1, migration); co(1, 3, migration); co(2, 4, migration); break;
      case 1: co(0, 0, migration); co(1, 2, migration); break;
      case 2: co(0, 5, migration); co(1, 6, migration); break;
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
    test();

    return 0;
}

 
#else

    std::cout << "Test requires the definition of MTL_HAS_PARMETIS (and of course"
	      << " the presence of ParMetis).\n";
    return 0;
}

#endif












