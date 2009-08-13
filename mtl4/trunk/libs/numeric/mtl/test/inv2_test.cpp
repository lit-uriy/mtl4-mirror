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
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;

// Alleged error reported by Dragan Vidovic

template<class ct>
mtl::dense2D<ct> inv2 ( const mtl::dense2D<ct> & M )
{
   mtl::dense2D<ct> N(2,2);
   ct d = M[0][0]*M[1][1]-M[0][1]*M[1][0];
   N[0][0] = M[1][1]/d;
   N[0][1] =-M[0][1]/d;
   N[1][0] =-M[1][0]/d;
   N[1][1] = M[0][0]/d;
   return N;
}


int test_main(int argc, char* argv[])
{
    typedef float ct;
    mtl::dense2D<ct> tmp(2, 2);
    tmp= 3, 5,
	 8, 9;

    mtl::dense2D<ct> tmp1 = inv2(tmp);
	
    return 0;
}
