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
#include <boost/numeric/mtl/operation/matrep.hpp>

using namespace std;
int test_main(int argc, char* argv[])
{
    using namespace mtl;
    unsigned size=3, row= size, col=size+1;

    dense2D<double>                         A(row, col), B(row*size,col*size);
    A= 0;

    A[0][0]=1;    A[0][1]=1;    A[0][2]=1; 
    A[1][0]=1;    A[1][1]=2;    A[1][2]=2;
    A[2][0]=9;    A[2][1]=3;    A[2][2]=2;
    A[2][3]=4;    A[0][3]=4;    A[1][3]=3;
    std::cout<<"A=\n"<< A <<"\n";
  
    B= matrep(A, size, size);
    std::cout<<"B=\n"<< B <<"\n";

    return 0;
}

