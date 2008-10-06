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

int test_main(int argc, char* argv[])
{
    using namespace std;
    using namespace mtl;
	
    dense2D<double> A(2,2), B(2,2);
    set_to_zero(B);
    A(0,0) = 1; A(0,1) = 2; A(1,0) = 3; A(1,1) = 4;
    B = A*A;
    
    cout << (A*A) << endl << B << endl;
    
    if (B(0, 1) != (A*A)(0,1))
	throw "Wrong value in matrix product expression!\n";

    return 0;
}
