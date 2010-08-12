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
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>



using std::cout; using std::complex;  

typedef complex<double> ct;


template <typename MatrixA, typename MatrixB, typename MatrixC>
void test(MatrixA&, MatrixB&, MatrixC&, const char* name)
{
    using mtl::matrix::inserter;
    MatrixA A(3, 4); 
    MatrixB B1(4, 5), B2(5, 6); 
    MatrixC C(3, 6);

    A= 0.0; B1= 0.0; B2= 0.0;

    {
	inserter<MatrixA>  A_ins(A);
	A_ins(0, 1) << 1.0;

	inserter<MatrixB>  B1_ins(B1);
	B1_ins(1, 2) << 1.0;

	inserter<MatrixB>  B2_ins(B2);
	B2_ins(2, 3) << 1.0;
    }

    //cout << "Input:\nA:\n" << A << "B1:\n" << B1 << "B2:\n" << B2;

    C= A * B1 * B2;

    cout << "\n\n" << name << "\n";
    cout << "C is:\n" << C << "\n";

    if (C[0][0] != 0.0)
	throw "C[0][0] should be 0!";

    if (C[0][3] != 1.0)
	throw "C[0][3] should be 1!";

    MatrixB I(6, 6);
    I = 1.0;

    cout << "Compute now C= A * B1 * B2 * I;\n";
    C= A * B1 * B2 * I;

    if (C[0][0] != 0.0)
	throw "C[0][0] should be 0!";

    if (C[0][3] != 1.0)
	throw "C[0][3] should be 1!";

    cout << "Compute now C= A * B1 * B2 * I * I * I * I;\n";
    C= A * B1 * B2 * I * I * I * I;

    if (C[0][0] != 0.0)
	throw "C[0][0] should be 0!";

    if (C[0][3] != 1.0)
	throw "C[0][3] should be 1!";

    cout << "Compute now C+= A * B1 * B2 * I;\n";
    C+= A * B1 * B2 * I;

    if (C[0][0] != 0.0)
	throw "C[0][0] should be 0!";

    if (C[0][3] != 2.0)
	throw "C[0][3] should be 2!";

    cout << "Compute now C-= A * B1 * B2 * I;\n";
    C-= A * B1 * B2 * I;

    if (C[0][0] != 0.0)
	throw "C[0][0] should be 0!";

    if (C[0][3] != 1.0)
	throw "C[0][3] should be 1!";
}



int test_main(int argc, char* argv[])
{
    using namespace mtl;

    unsigned size= 7; 
    if (argc > 1) size= atoi(argv[1]); 

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppled_2_row_mask>  d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);


    test(dr, dr, dr, "Dense row major");
    test(dc, dr, dr, "Dense column major as product of dense rows");
    test(dc, dr, dc, "Dense column major as product of dense rows and column");

    test(mzd, mzd, mzd, "Morton Z-order");
    test(d2r, mzd, d2r, "Hybrid 2 row-major * Morton Z-order");

    test(cr, cr, cr, "Compressed row major");
    test(cc, cr, cc, "Compressed column major * row");

    test(drc, drc, drc, "Dense row major complex");
    test(drc, dc, drc, "Dense row major complex * column double");
    test(crc, crc, crc, "Compressed row major complex");
    test(crc, dc, crc, "Compressed row major complex * dense column major double");

    return 0;
}
