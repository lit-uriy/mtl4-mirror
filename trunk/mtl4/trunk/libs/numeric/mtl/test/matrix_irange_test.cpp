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
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/recursion/matrix_recursator.hpp>

using namespace mtl;
using namespace std;  

template <typename Size>
struct ss
{
    int operator[](Size i) { cout << "Size\n"; return 1; }
    int operator[](mtl::irange i) { cout << "irange\n"; return 2;}

};

void f(std::size_t i)
{
    cout << "with size_t\n";
}

void f(irange)
{
    cout << "with irange\n";
}


template <typename Matrix>
void test(Matrix& A, const char* name)
{
#if 0  // Still working on it
    f(1);
    f(irange(1, 2));

    ss<std::size_t> s;
    s[3];
    s[irange(1, 2)];

    A= 0.0;
    //A[1][1]= 1.0; 
    hessian_setup(A, 1.0);

    std::cout << "\n" << name << "\nA == \n" << A;
    
    cout << "A[irange(1, 4)][irange(1, imax)] == \n" 
	 << A[irange(1, 4)][irange(1, imax)] << "\n";

    Matrix B(A[irange(1, 4)][irange(1, imax)]);
    if (B[1][1] != 4.0) throw "Wrong value in B";

    if (A[irange(1, 4)][irange(1, imax)][1][1] != 4.0) throw "Wrong value in A[][]";

    Matrix C(A[irange(3, imax)][irange(2, 3)]);
    if (C[0][1] != 5.0) throw "Wrong value in C";

    cout << "A[irange(3, imax)][irange(2, 3)] == \n" 
	 << A[irange(3, imax)][irange(2, 3)];
#endif
}


int test_main(int argc, char* argv[])
{
    const unsigned size= 5; 

    dense2D<double> dc(size, size-2);
    dense2D<double, matrix::parameters<col_major> >  dcc(size, size-2);
    dense2D<float>                                   fc(size, size-2);
    morton_dense<double,  morton_mask>               mdc(size, size-2);
    morton_dense<double, doppled_32_col_mask>        mcc(size, size-2);

    test(dc, "dense2D");
#if 0
    test(dcc, "dense2D col-major");
    test(mdc, "pure Morton");
    test(mcc, "Hybrid col-major");
#endif
    return 0;
}
