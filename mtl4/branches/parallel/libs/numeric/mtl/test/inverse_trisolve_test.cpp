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
    A.change_dim(5, 5); A= 0.0;
    {
	mtl::matrix::inserter<Matrix>   ins(A);
	ins[0][0] << 7; ins[1][1] << 8; ins[1][3] << 2; ins[1][4] << 3;
	ins[2][2] << 2; ins[3][3] << 4; ins[4][4] << 9;
    }
    Matrix B(trans(A));
    
    double xa[] = {1, 2, 3, 4, 5};
    mtl::dense_vector<double> x(xa), b;
    
    b= A * x;
    x= 0.0;
    
    cout << name << "\nA = \n" << A << "b = " << b << "\n";

    invert_diagonal(A);
    
    x= upper_trisolve(A, b, mtl::tag::inverse_diagonal());
    cout << "x = upper_trisolve(A, b) ==" << x << "\n\n";
    if (std::abs(x[2] - 3.0) > 0.0001) throw "Wrong result in upper_trisolve!";


    x= xa;
    
    b= B * x;
    x= 0.0;
    
    cout << "B = \n" << B << "b = " << b << "\n";
	
    invert_diagonal(B);

    x= lower_trisolve(B, b, mtl::tag::inverse_diagonal());
    cout << "x = lower_trisolve(B, b) ==" << x << "\n\n";
    if (std::abs(x[2] - 3.0) > 0.0001) throw "Wrong result in lower_trisolve!";


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

    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(mzd, "Morton Z-order");
    test(d2r, "Hybrid 2 row-major");
    test(cr, "Compressed row major");
    test(cc, "Compressed column major");

    return 0;
}
