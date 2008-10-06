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
    
    value_type array[][3]= {{3, 7.2, 0}, {2, 4.444, 5}};
    A= array;

    std::cout << "\n" << name << ", assignment: A = \n" << A << "\n";

    if (num_rows(A) != 2 || num_cols(A) != 3)
	throw "Wrong matrix size";
    if (A[1][0] != value_type(2))
	throw "Wrong value inserted";

    Matrix B(array);

    std::cout << "\n" << name << ", construction: B = \n" << B << "\n";

    if (num_rows(B) != 2 || num_cols(B) != 3)
	throw "Wrong matrix size";
    if (B[1][0] != value_type(2))
	throw "Wrong value inserted";
}


int test_main(int argc, char* argv[])
{
    
    dense2D<double>                                      dr;
    dense2D<double, matrix::parameters<col_major> >      dc;
    morton_dense<double, recursion::morton_z_mask>       mzd;
    morton_dense<double, recursion::doppled_2_row_mask>  d2r;
    compressed2D<double>                                 cr;
    compressed2D<double, matrix::parameters<col_major> > cc;

    dense2D<complex<double> >                            drc;
    compressed2D<complex<double> >                       crc;

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
