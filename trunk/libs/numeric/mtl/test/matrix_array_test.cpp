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
    using mtl::dense2D; using mtl::dense_vector;
	
    dense2D<double> *A;
    A = new dense2D<double>[2];
    A[0] = dense2D<double>(3,3);
    A[1] = dense2D<double>(4,4);
    cout << A[0] << endl;
    cout << A[1] << endl;
    
    dense_vector<double> *x;
    x = new dense_vector<double>[2];
    x[0] = dense_vector<double>(3);
    x[1] = dense_vector<double>(4);
    // x[1] = dense_vector<double>(5); // Must throw an exception because 
    cout << x[0] << endl;
    cout << x[1] << endl;


    return 0;
}
