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

int test_main(int argc, char* argv[])
{
    using namespace std;
    using namespace mtl;
    

    // Define a 6x5 sparse matrix in a 3x3 block-sparse
    typedef dense2D<double>    m_t;
    typedef compressed2D<m_t>  matrix_t;
    matrix_t                   A(3, 3);
    {
	matrix::inserter<matrix_t> ins(A);

	// First block
	m_t  b1(1, 1);
	b1(0, 0)= 1.0;
	ins(0, 2) << b1;

	// Second block
	m_t  b2(2, 3);
	b2=       0.0;
	b2[0][1]= 2.0;
	b2[1][2]= 3.0;
	ins(1, 0) << b2;

	m_t b3(3, 1);
	b3= 0.0;
	b3[1][0]= 4.0;
	ins(2, 1) << b3;
    }
    cout << "A is " << A << endl; // works but is completely unreadable

    /* Should be something like this:

    [                   [ 1]] // b1
    [[  0   2   0]          ] // b2
    [[  0   0   3]          ]
    [             [  0]     ] // b3
    [             [  4]     ]
    [             [  0]     ]

    */

    // Access blocks (they are read-only) for sparse matrices
    cout << "The block A(1, 0) is \n" << A(1, 0) << endl;
    cout << "The block A[1][0] is \n" << A[1][0] << endl;

    // Access elements in blocks 
    cout << "In block A(1, 0), the element (0, 1) is " << A(1, 0)(0, 1) << endl;
    cout << "In block A(1, 0), the element [0][1] is " << A(1, 0)[0][1] << endl;
    cout << "In block A[1][0], the element [0][1] is " << A[1][0][0][1] << endl << endl;


    typedef dense_vector<double> v_t;
    typedef dense_vector<v_t>    vector_t;
    vector_t                     x(3), y(3);

    // x= [[0, 5, 3], [1], [8]]^T
    x[0]= v_t(3, 0.0); x[0][1]= 5.0; x[0][2]= 3.0; // first block of x = [0, 5, 3]^T
    x[1]= v_t(1, 1.0);                             // second block of x 
    x[2]= v_t(1, 8.0);                             // third block

    cout << "x is " << x << endl; 

    // For y we would only need the vector sizes [[?], [?, ?], [?, ?, ?]]^T
    // To avoid valgrind complains we set to 0
    y[0]= v_t(1, 0.0); y[1]= v_t(2, 0.0); y[2]= v_t(3, 0.0);

    cout << "y is " << y << endl; 

    // Block-sparse matrix * blocked vector !!!
    y= A*x;

    cout << "y after multiplication is " << y << endl
	 << "Should be [[8], [10, 9], [0, 4, 0]]^T." << endl; 

    return 0;
}
