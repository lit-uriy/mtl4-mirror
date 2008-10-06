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


using namespace std;
using namespace mtl;


template <typename Matrix>
void test(Matrix& A, const char* name)
{
    typedef typename Collection<Matrix>::value_type   value_type;
    cout << "\n" << name << "\n" << "A =\n" << A;

    cout << "Pointer to major (row) index array is " << A.address_major() 
	 << ", pointer to minor (column) index array is " << A.address_minor() 
	 << ", pointer to data array is " << A.address_data() << "\n";
}


int test_main(int argc, char* argv[])
{
    compressed2D<double>                                 cr(2, 2);
    cr= 2.0;

    test(cr, "Compressed row major");
    return 0;
}
