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


template <typename Matrix>
void test(Matrix& A, const char* name)
{
    typedef typename mtl::Collection<Matrix>::value_type   value_type;

    A= 3, 7.2, 0,
       2, 4.444, 5;

    std::cout << "\n" << name << ", assignment: A = \n" << A << "\n";

    if (A[1][0] != value_type(2))
	throw "Wrong value inserted";
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    dense2D<double>                                      dr(2, 3);
    dense2D<double, matrix::parameters<col_major> >      dc(2, 3);
    morton_dense<double, recursion::morton_z_mask>       mzd(2, 3);
    morton_dense<double, recursion::doppled_2_row_mask>  d2r(2, 3);
    compressed2D<double>                                 cr(2, 3);
    compressed2D<double, matrix::parameters<col_major> > cc(2, 3);

    dense2D<complex<double> >                            drc(2, 3);
    compressed2D<complex<double> >                       crc(2, 3);

    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(mzd, "Morton Z-order");
    test(d2r, "Hybrid 2 row-major");
    test(cr, "Compressed row major");
    test(cc, "Compressed column major");
    test(drc, "Dense row major complex");
    test(crc, "Compressed row major complex");

	
    return 0;
}
