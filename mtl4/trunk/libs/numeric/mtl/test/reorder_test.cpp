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
    
    value_type array[][3]= {{1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.}};
    A= array;

    cout << "\n" << name << "\n" << "A =\n" << A;

    int reordering[]= {2, 1};
    typename matrix::traits::reorder<>::type  R= matrix::reorder(reordering);
    cout << "\nR =\n" << R;    

    Matrix B= R * A;
    cout << "\nB= R * A =\n" << B;
    
    if (B[1][0] != value_type(4.)) throw "Wrong value after row reordering!";

    Matrix B2= B * trans(R);
    cout << "\nB * trans(R) =\n" << B2;
    
    if (B2[1][0] != value_type(6.)) throw "Wrong value after column reordering!";    
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
