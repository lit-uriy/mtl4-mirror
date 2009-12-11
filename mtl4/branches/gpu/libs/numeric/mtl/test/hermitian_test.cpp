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



int test_main(int argc, char* argv[])
{

    mtl::dense2D<double> A(3,3);
    mtl::dense2D<double> B(3,3);
    mtl::dense2D<double> C(3,3);
    A = 5;
    B = A;
    // C = hermitian(B) * A;
    // #warning "Test temporarily disabled."    

    return 0;
}
