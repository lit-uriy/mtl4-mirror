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
void check(const Matrix& A, int begin, int end)
{
    typedef typename Collection<Matrix>::value_type   value_type;

    for (int i= 0; i < num_rows(A); i++)
	for (int j= 0; j < num_cols(A); j++) {
	    int band= j - i;
	    if (band < begin) {
		if (A[i][j] != value_type(0))
		    throw "Value must be zero left of the bands";
	    } else if (band >= end) {
		if (A[i][j] != value_type(0))
		    throw "Value must be zero right of the bands";
	    } else
		if (A[i][j] != value_type(band + 0))
		    throw "Wrong non-zero value within the bands";
	}
}


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
    Matrix B= bands(A, 2, 4);
    cout << "\nbands(A, 2, 4) = \n" << B;
    check(B, 2, 4);

    Matrix U= upper(A);
    cout << "\nupper(A) = \n" << U;
    check(U, 0, 10000);

    Matrix SU= strict_upper(A);
    cout << "\nstrict_upper(A) = \n" << SU;
    check(SU, 1, 10000);

    Matrix L= lower(A);
    cout << "\nlower(A) = \n" << L;
    check(L, -10000, 1);
    
    Matrix SL= strict_lower(A);
    cout << "\nstrict_lower(A) = \n" << SL;
    check(SL, -10000, 0);
    
    Matrix P= trans(A) * upper(A), P_cmp= trans(A) * U, P_cmp2= trans(A) * A;
    cout << "\ntrans(A) * upper(A) = \n" << with_format(P, 4, 3);
    // cout << " for comparison trans(A) * U = \n" << with_format(P_cmp, 4, 3);
    // cout << " for comparison trans(A) * A = \n" << with_format(P_cmp2, 4, 3);
    if (abs(P[1][1] - P_cmp[1][1]) > .00001) throw "Multiplication wrong";

#if 0
    // Take this out as sparse matrices have no sub-matrix; only for debugging wrong products
    Matrix A2= sub_matrix(A, 0, 3, 0, 3), U2= upper(A2);
    Matrix P2= A2 * upper(A2); //, P2_cmp= A2 * U2, A2_square= A2 * A2;
    cout << "\nA2 * upper(A2) = \n" << with_format(P2, 4, 3);
    cout << " for comparison A2 * U = \n" << with_format(P2_cmp, 4, 3);
    cout << " for comparison A2 * A2 = \n" << with_format(A2_square, 4, 3);
    if (abs(P2[1][1] - P2_cmp[1][1]) > .00001) throw "Multiplication wrong";
#endif


#if 0
    // Would too painfully slow !

    Matrix D= diagonal(A);
    cout << "\ndiagonal(A) = \n" << D;
    check(D, 0, 1);

    Matrix T= tri_diagonal(A);
    cout << "\ntri_diagonal(A) = \n" << T;
    check(T, 0, 1);
#endif
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
