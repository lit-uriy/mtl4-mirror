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

// Regression test for problem reported by Garth Wells

#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

namespace mtl {
    using mtl::vector::num_rows; // has no effect with the friend definition
}


int test_main(int argc, char* argv[])
{
    mtl::dense_vector<double> x(10);
    unsigned int size1 = num_rows(x);
    unsigned int size2 = mtl::num_rows(x);              // does not compile with friend definition
    unsigned int size3 = mtl::vector::num_rows(x);      // does not compile with friend definition either
}
