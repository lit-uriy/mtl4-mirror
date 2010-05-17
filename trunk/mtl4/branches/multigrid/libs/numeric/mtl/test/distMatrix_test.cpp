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
#include <boost/numeric/mtl/operation/distMatrix.hpp>

using namespace std;
int test_main(int argc, char* argv[])
{
    using namespace mtl;
    
    dense2D<double>                         A(4, 3), B(2, 3), C(4, 2), quadA(4, 3);
    A= 0; B= 0;

    A[0][0]=1;    A[0][1]=2;    A[0][2]=3; 
    A[1][0]=4;    A[1][1]=5;    A[1][2]=6;
    A[2][0]=2;    A[2][1]=4;    A[2][2]=6;
    A[3][0]=1;    A[3][1]=2;    A[3][2]=3;
    std::cout<<"A=\n"<< A <<"\n";
    
    B[0][0]=4;    B[0][1]=5;    B[0][2]=1; 
    B[1][0]=6;    B[1][1]=2;    B[1][2]=0;
    std::cout<<"B=\n"<< B <<"\n";

    C= distMatrix(A, B);
    std::cout<<"Dist(A,B)=\n"<< C <<"\n";

//     quadA=quadMatrix(A);
//     std::cout<<"quadA=\n"<< quadA <<"\n";

    return 0;
}
