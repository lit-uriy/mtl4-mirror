// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

// Currently needs -DMTL_DEEP_COPY_CONSTRUCTOR !!!

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int test_main(int argc, char* argv[])
{
    using namespace std;
    using namespace mtl;
    
    compressed2D<double> m(3, 3);
    {
	matrix::inserter<compressed2D<double> > ins(m);
	ins(0, 1) << 2.0; ins(1, 0) << 1.0;
	ins(1, 1) << 4.0; ins(2, 2) << 5.0;
    }

    dense_vector<double> x(3), y(3);
    for (int i= 0; i < size(x); i++) x[i]= double(i+1);

    y = trans(m) * x;
    cout << y << '\n';

    if (y[0] != 2.0) throw "y[0] should be 2.0!\n";

    return 0;
}
