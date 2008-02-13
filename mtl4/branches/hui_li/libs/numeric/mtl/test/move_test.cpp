// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <adobe/move.hpp>


// Everything in the test is double
// How to test sparse generically? 

using namespace std;
using namespace mtl;
	

// Return a matrix with move semantics
// Return also the address of the first entry to be sure that it is really moved
template <typename Matrix>
Matrix f(const Matrix&, double*& a00)
{
    Matrix A(3, 3);
    A= 5.0;
    a00= &A[0][0];
    return adobe::move(A);
}

template <typename Matrix>
void test(const Matrix&)
{
    double *p;
    Matrix A(3, 3);
    A= 0.0;
   
    A= f(A, p);

    if (A[0][0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (&A[0][0] != p) 
	throw "Matrix is not moved but copied!";
#if 0
    Matrix B= f(A, p);

    if (B[0][0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (&B[0][0] != p) 
	throw "Matrix is not moved but copied!";

    // This type is guarateed to be different to f's return type
    // In this case the matrix MUST be copied
    morton_dense<double, recursion::doppled_2_row_mask> C(3, 3);

    C= f(A, p);

    if (C[0][0] != 5.0) 
	throw "Wrong value trying to move, should be 5.0!";
    if (&C[0][0] == p) 
	throw "Matrix be copied not moved!";
#endif
}




int test_main(int argc, char* argv[])
{
    dense2D<double>                                      dr(3, 3);
#if 0
    dense2D<double, matrix::parameters<col_major> >      dc(3, 3);
    morton_dense<double, recursion::morton_z_mask>       mzd(3, 3);
#endif

    return 0;
}
