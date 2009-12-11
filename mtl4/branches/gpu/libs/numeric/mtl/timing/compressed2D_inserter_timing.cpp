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
#include <boost/timer.hpp>


using namespace std;

void test(int n)
{
    mtl::compressed2D<double> A(n*n, n*n);
    boost::timer start;
    {
	mtl::matrix::inserter<mtl::compressed2D<double> > ins(A);
	for (int i= 0; i < n; i++) {
	    int r= i * n, re= r+n;
	    if (i > 0)
		for (int j= r; j < re; j++)
		    ins[j][j-n] << -1.0;
	    ins[r][r] << 4.0;
	    ins[r][r+1] << -1;
	    for (int j= r+1; j < re-1; j++) {
		ins[j][j-1] << -1.0;
		ins[j][j] << 4.0;
		ins[j][j+1] << -1.0;
	    }
	    ins[re-1][re-2] << -1.0;
	    ins[re-1][re-1] << 4.0;
	    if (i < n-1)
		for (int j= r; j < re; j++)
		    ins[j][j+n] << -1.0;
	}
    }
    //cout << "A is \n" << A;
    cout << "Insertion took " << start.elapsed() << '\n';
	
}

 
int test_main(int argc, char* argv[])
{
    test(atoi(argv[1]));

    return 0;
}
