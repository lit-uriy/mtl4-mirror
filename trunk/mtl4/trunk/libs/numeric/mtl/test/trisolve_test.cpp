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


using namespace std;
using namespace mtl;

template <typename Matrix>
void test(Matrix& A, const char* name)
{
    typedef typename Collection<Matrix>::value_type   value_type;
    A.change_dim(5, 5);
	{
		matrix::inserter<Matrix>   ins(A);
		ins[0][0] << 7; ins[1][1] << 8; ins[1][3] << 2; ins[1][4] << 3;
		ins[2][2] << 2; ins[3][3] << 4; ins[4][4] << 9;
	}

	double xa[] = {1, 2, 3, 4, 5};
	dense_vector<double> x(xa), b;

	b= A * x;
	x= 0.0;

	cout << "A = \n" << A << "b = " << b << "\n";
	
	// x= trisolve(A, b);
	cout << "x = trisolve(A, b) ==" << x << "\n";
	
	x= xa;
	Matrix B(trans(A));

	b= B * x;
	x= 0.0;

	cout << "B = \n" << B << "b = " << b << "\n";
	
	// x= trisolve(B, b);
	cout << "x = trisolve(B, b) ==" << x << "\n";

}

int test_main(int argc, char* argv[])
{

    compressed2D<double>                                 cr;
    compressed2D<double, matrix::parameters<col_major> > cc;

	test(cr, "Compressed row major");
    test(cc, "Compressed column major");

    return 0;
}
