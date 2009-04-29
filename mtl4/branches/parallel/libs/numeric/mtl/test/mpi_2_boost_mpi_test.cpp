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



using namespace std;  

#include <boost/mpi.hpp>
#include <iostream>
#include <boost/serialization/string.hpp>
#include <boost/test/minimal.hpp>

namespace mpi = boost::mpi;


int test_main(int argc, char* argv[]) 
{
  mpi::environment env(argc, argv);
  mpi::communicator world;

  if (world.rank() == 0) {
    for (int i= 1; i < world.size(); i++)
	world.send(i, 0, std::string("Hello"));
    std::string msg;
    world.recv(1, 1, msg);
    std::cout << msg << "!" << std::endl;
  } else {
    std::string msg;
    world.recv(0, 0, msg);
    std::cout << msg << ", ";
    std::cout.flush();
    if (world.rank() == 1)
	world.send(0, 1, std::string("world"));
  }

  return 0;
}

 














