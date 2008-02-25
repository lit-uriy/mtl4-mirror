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
    //return adobe::move(A);
    return A;
}

template <typename Matrix>
void print(const Matrix& matrix, double*& p)
{
    cout << "Data was " << (&matrix.data[0] == p ? "moved.\n" : "copied.\n");
}

template <typename Matrix>
void test(const Matrix&, const char* text)
{
    cout << '\n' << text << '\n';

    double *p;
    Matrix A(3, 3);
    A= 0.0;
   
    cout << "A= f(A, p);\n";
    A= f(A, p);
    print(A, p);

    if (A[0][0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (&A[0][0] != p) 
	throw "Matrix is not moved but copied!";
#if 0
    cout << "Matrix B= f(A, p);\n";
    Matrix B= f(A, p);
    print(B, p);

    if (B[0][0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (&B[0][0] != p) 
	throw "Matrix is not moved but copied!";

    // This type is guarateed to be different to f's return type
    // In this case the matrix MUST be copied
    morton_dense<double, recursion::doppled_2_row_mask> C(3, 3);

    cout << "C= f(A, p);  // C and A have different types\n";
    C= f(A, p);

    if (C[0][0] != 5.0) 
	throw "Wrong value trying to move, should be 5.0!";
    if (&C[0][0] == p) 
	throw "Matrix must be copied not moved!";


    // Other matrix type, in this case the matrix MUST be copied
    morton_dense<double, recursion::morton_mask>   D(A);

    cout << "D(A);  // C and A have different types\n";
    print(D, &A.data[0][0]);

    if (D.data[0] != 5.0) 
	throw "Wrong value in copy constructor, should be 5.0!";
    if (&D.data[0] == &A.data[0]) 
	throw "Matrix must be copied not moved!";

    Matrix E= sub_matrix(A, 0, 1, 0, 1);

    cout << "Matrix E= sub_matrix(A, 0, 1, 0, 1);\n";
    print(E, &A.data[0][0]);

    if (&E.data[0][0] != &A.data[0][0]) 
	throw "Sub-matrix must be referred to not copied!";

    cout << "E= sub_matrix(A, 1, 2, 1, 2);\n";
    E= sub_matrix(A, 1, 2, 1, 2);    
    print(E, &A.data[1][1]);

    if (&E.data[0][0] == &A.data[1][1]) 
	throw "Matrix must be copied not referred to!";

    Matrix F= clone(sub_matrix(A, 0, 1, 0, 1));

    cout << "Matrix F= clone(sub_matrix(A, 0, 1, 0, 1));\n";
    print(F, &A.data[0][0]);

    if (&F.data[0][0] == &A.data[0][0]) 
	throw "Sub-matrix must be forced to copy!";
#endif
}




int test_main(int argc, char* argv[])
{
    dense2D<double>                                      dr(3, 3);
    dense2D<double, matrix::parameters<col_major> >      dc(3, 3);
    morton_dense<double, recursion::morton_z_mask>       mzd(3, 3);

    test(dr, "Dense matrix");
#if 0
    test(dc, "Column-major dense matrix");
    test(mzd, "Morton-order z-mask");

    compressed2D<double>                                 crs(3, 3);
    compressed2D<double, matrix::parameters<col_major> > ccs(3, 3);

    sparse_test(crs, "CRS");
    sparse_test(ccs, "CCS");

#endif

    return 0;
}
