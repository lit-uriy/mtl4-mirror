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

// Currently needs -DMTL_DEEP_COPY_CONSTRUCTOR !!!

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int test_main(int argc, char* argv[])
{
    using namespace std;
    
    
    mtl::compressed2D<double> res(3, 3), m(3, 3);
    {
	mtl::matrix::inserter<mtl::compressed2D<double> > ins(m);
	ins(0, 1) << 2.0; ins(1, 0) << 1.0;
	ins(1, 1) << 4.0; ins(2, 2) << 5.0;
    }

    res = trans(m)*m;
    cout << res;

    if (res[0][1] != 4.0) throw "res[0][1] should be 4.0!\n";

    return 0;
}
