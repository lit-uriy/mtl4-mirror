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
#include <string>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/recursion/matrix_recursator.hpp>
 

using namespace std;  

std::string program_dir; // Ugly global variable !!!

template <typename Matrix>
void test(Matrix& A, const char* name)
{
    typedef typename mtl::Collection<Matrix>::value_type vt;
    A= vt(0);
    {
	mtl::matrix::inserter<Matrix> ins(A);
	ins[0][1] << vt(2);
	ins[2][0] << vt(3);
    }

    std::cout << "\n" << name << "\n";
    string fname( mtl::io::join( program_dir, string("matrix_market/write_test_1_") + string(name) + string(".mtx") ) );
    cout << "File name is " << fname << "\nA is\n" << A;
    
    mtl::io::matrix_market_ostream oms(fname);
    oms << A;
    oms.close();

    Matrix B, C; // (mtl::io::matrix_market(fname)), C;
    B= mtl::io::matrix_market(fname);
    cout << "\nRead back (after <<) results in\n" << B;
    if (num_rows(B) != 4 || num_cols(B) != 3) throw "B has wrong dimension.";
    if (B[0][2] != vt(0)) throw "B[0][2] should be 0!";
    assert (B[0][1] == vt(2));
    if (B[0][1] != vt(2)) throw "B[0][1] should be 2!";

#if 0
    fname[25]= '2';
    mtl::io::matrix_market(fname)= A;
    C= mtl::io::matrix_market(fname);

    cout << "\nRead back (after assignment) results in\n" << C;
    if (num_rows(C) != 4 || num_cols(C) != 3) throw "C has wrong dimension.";
    if (C[0][2] != vt(0)) throw "C[0][2] should be 0!";
    if (C[0][1] != vt(2)) throw "C[0][1] should be 2!";
#endif
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;

    compressed2D<double>                             cdr(4, 3);
    compressed2D<int>                                cir(4, 3);
    compressed2D<std::complex<double> >              ccr(4, 3);
    compressed2D<double, matrix::parameters<col_major> >    cdc(4, 3);
    dense2D<double>                                  dc(4, 3);
    dense2D<int>                                     di(4, 3);
    dense2D<double, matrix::parameters<col_major> >  ddc(4, 3);
    dense2D<std::complex<double>, matrix::parameters<col_major> >  dcc(4, 3);
    dense2D<float>                                   fc(4, 3);
    morton_dense<double,  morton_mask>               mdc(4, 3);
    morton_dense<double, doppled_32_col_mask>        mcc(4, 3);

    program_dir= mtl::io::directory_name(argv[0]);
    test(cdr, "compressed2D_double");
    test(cir, "compressed2D_int");
    test(ccr, "compressed2D_complex");
    test(cdc, "compressed2D_double_col-major");
    test(dc, "dense2D_double");
    test(di, "dense2D_int");
    test(ddc, "dense2D_double_col-major");
    test(dcc, "dense2D_complex_col-major");
    test(mdc, "pure_Morton_double");
    test(mcc, "Hybrid_col-major_double");

    return 0;
}
