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
void add_row(Matrix& A, typename mtl::Collection<Matrix>::size_type n= 1)
{
    Matrix A_tmp(num_rows(A) + n, num_cols(A));
    sub_matrix(A_tmp, 0, num_rows(A), 0, num_cols(A))= A;
    swap(A_tmp, A);
}

template <typename Matrix>
void test(Matrix& A, const char* name)
{
    typedef typename mtl::Collection<Matrix>::value_type   value_type;

    A.change_dim(3, 3);
    {
	mtl::matrix::inserter<Matrix>   ins(A);
	for (int i= 0; i < num_rows(A); i++)
	    for (int j= 0; j < num_cols(A); j++)
		ins[i][j]= value_type(j - i + 0);
    }
    cout << "\n" << name << "\n" << "A =\n" << A;

    add_row(A);
    cout << "\nA after adding one row A =\n" << A;

    if (num_rows(A) != 4 || num_cols(A) != 3)
	throw "Wrong dimension after adding one row";

    if (A[2][1] != value_type(-1))
	throw "Wrong value in A[2][1] after adding one row";

    add_row(A, 2);
    cout << "\nA after adding two rows A =\n" << A;

    if (num_rows(A) != 6 || num_cols(A) != 3)
	throw "Wrong dimension after adding two rows";
    
    if (A[2][1] != value_type(-1))
	throw "Wrong value in A[2][1] after adding two rows";
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    
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
    // test(cr, "Compressed row major");
    test(drc, "Dense row major complex");
    // test(crc, "Compressed row major complex");

    // For better readability I don't want finish with a complex
    // test(cc, "Compressed column major");

    return 0;
}
