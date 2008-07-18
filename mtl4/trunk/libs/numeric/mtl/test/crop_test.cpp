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

    A.change_dim(6, 5);
    {
	matrix::inserter<Matrix>   ins(A);
	for (int i= 0; i < num_rows(A); i++)
	    for (int j= 0; j < num_cols(A); j++)
		ins[i][j]= value_type(j - i + 0);
    }
    cout << "\n" << name << "\n" << "A =\n" << A;

    cout << "Number of non-zeros: " << A.nnz() << '\n';
    crop(A);
    cout << "Number of non-zeros after crop: " << A.nnz() << '\n';

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
    test(drc, "Dense row major complex");
    test(crc, "Compressed row major complex");

    // For better readability I don't want finish with a complex
    test(cc, "Compressed column major");

    return 0;
}
