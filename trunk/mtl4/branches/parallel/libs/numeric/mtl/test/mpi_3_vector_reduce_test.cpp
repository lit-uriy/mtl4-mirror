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

#if defined(MTL_HAS_MPI)

#include <boost/mpi.hpp>
#include <cstdlib>
#include <boost/numeric/mtl/mtl.hpp>

namespace mpi = boost::mpi;

template <typename Vector>
struct vector_plus
{
    Vector operator()(const Vector& x, const Vector& y)
    {
	return Vector(x + y);  // because implicit conversion from expression template to vector is disabled
    }
};

int main(int argc, char* argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;

    std::srand(time(0) + world.rank());
    int my_number = std::rand();

    typedef mtl::dense_vector<double>      vector_type;
    typedef vector_plus<vector_type>       plus;

    vector_type vrand(3), sum(3);
    random(vrand);

    if (world.rank() == 0) {
	reduce(world, vrand, sum, plus(), 0);
	std::cout << "The sum of all vectors is " << sum << std::endl;
    } else {
	reduce(world, vrand, plus(), 0);
    }  
    return 0;
}

#else 

int main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_MPI (and of course"
	      << " the presence of MPI).\n";
    return 0;
}

#endif
 
