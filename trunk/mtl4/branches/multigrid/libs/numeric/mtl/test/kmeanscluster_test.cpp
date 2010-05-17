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
#include <boost/utility.hpp>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/operation/kmeanscluster.hpp>

using namespace std;
int test_main(int argc, char* argv[])
{
    using namespace mtl;
    
    dense2D<double>                         A(4, 2), B(4, 3);
    A= 0; B= 0;

    A[0][0]=1;    A[0][1]=1; 
    A[1][0]=2;    A[1][1]=1;
    A[2][0]=4;    A[2][1]=3;
    A[3][0]=5;    A[3][1]=4;
    
    B= kmeanscluster(A, 2);
    std::cout<<"kmeanscluster(A,2)=\n"<< B <<"\n";

    return 0;
}
