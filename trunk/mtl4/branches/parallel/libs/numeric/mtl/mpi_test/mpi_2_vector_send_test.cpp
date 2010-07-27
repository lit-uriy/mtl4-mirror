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

struct ss 
{
    ss() : v(0) {}
    ss(int v) : v(v) {}
    operator int() { return v; }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned version) { ar & v; }

    friend std::ostream& operator<<(std::ostream& os, ss x) { return os << x.v; }

    int v;
};

template <typename T>
void test()
{
    mpi::communicator world;

    mtl::dense_vector<T> v(world.rank() ? 0 : 2);

    if (world.rank() == 0) {
	v[0]= 4 - world.rank(), v[1]= 9;
	world.send(1, 0, v);
    } else {
	std::cout << "v on proc 1 is " << v << std::endl;
	world.recv(0, 0, v);
	std::cout << "v on proc 1 is " << v << std::endl;
	if (v[0] != 4) throw "v[0] should be 4!";
    }
}


int test_main(int argc, char* argv[]) 
{
    mpi::environment env(argc, argv);
    mpi::communicator world;

    if (world.size() != 2) {
	std::cerr << "Example works only for 2 processors!\n";
	env.abort(87);
    }

    test<int>();
    test<ss>();

    return 0;
}

 














