// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
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
 
using namespace mtl;
using namespace std;  


template <typename Matrix>
void test_file(Matrix& A, const char* file_name, const char* comment)
{
    mtl::io::matrix_market_ifstream ms(file_name);
    ms >> A;
    std::cout << "Read from " << file_name << " (" << comment << ") is " 
	      << num_rows(A) << "x" << num_cols(A) << "\n";

    if (num_rows(A) > 9 && num_cols(A) > 9) {
	int reordering[]= {0, 1, 2, 3, 4, 5, 6, 7, 8};
	matrix::traits::reorder<>::type  R= matrix::reorder(reordering, num_cols(A)),
	    R2= matrix::reorder(reordering, num_rows(A));
	Matrix B(R * A * trans(R2));
	std::cout << "A[0:9][0:9] is:\n" << with_format(B, 8, 3);
    } else
	std::cout << "A is:\n" << with_format(A, 8, 3);
}


template <typename Matrix>
void test(Matrix& A, const char* name)
{
    std::cout << "\n" << name << "\n";

    test_file(A, "matrix_market/jgl009.mtx", "general pattern"); 
    test_file(A, "matrix_market/mhd1280b.mtx", "Hermitian"); 
    // test_file(A, "matrix_market/plskz362.mtx", "Skew-symmetric"); // has only 0s in A[:9][:9]
    test_file(A, "matrix_market/bcsstk01.mtx", "Real symmetric");

    Matrix B(io::matrix_market("matrix_market/jgl009.mtx")), C;
    std::cout << "Matrix market file read in constructor:\n" << B;

    C= io::matrix_market("matrix_market/jgl009.mtx");
    std::cout << "Matrix market file assigned:\n" << B;
    
}


int test_main(int argc, char* argv[])
{
    const unsigned size= 5; 

    compressed2D<double>                             cdc;
    compressed2D<std::complex<double> >              ccc;
    dense2D<double>                                  dc;
    dense2D<double, matrix::parameters<col_major> >  dcc;
    dense2D<float>                                   fc;
    morton_dense<double,  morton_mask>               mdc;
    morton_dense<double, doppled_32_col_mask>        mcc;

    // This is an ugly test to be removed
    if (strlen(argv[0]) > strlen("matrix_market_test")+4) {
	std::cerr << "For simplicity this test works only in the test directory\n"
		  << "Please cd there and rerun the test.";
	return 0;
    }

    test(cdc, "compressed2D");
    test(ccc, "compressed2D complex");
    test(dc, "dense2D");
    test(dcc, "dense2D col-major");
    test(mdc, "pure Morton");
    test(mcc, "Hybrid col-major");

    return 0;
}
