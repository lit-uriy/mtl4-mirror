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


template <typename Matrix>
void test(Matrix& A, const char* name)
{
    A= 0.0;
    A[0][0]= 1.0; 
    hessian_setup(A, 1.0);

    matrix::recursator<Matrix> rec(A), nw= north_west(rec);

    std::cout << "\n" << name << "\n";
    std::cout << "north_west:\n" << *nw << '\n';    

    std::cout << "north_west of north_west:\n" << *north_west(nw) << '\n';
    if ((*north_west(nw))[0][0] != 0.0) throw "(*north_west(nw))[0][0] != 0.0";
    (*north_west(nw))[0][0]= 2.0;

    std::cout << "south_east of north_west:\n" << *south_east(nw) << '\n';
    if ((*south_east(nw))[0][0] != 4.0) throw "(*south_east(nw))[0][0] != 4.0";

    std::cout << "north_west of north_west:\n" << *north_west(nw) << '\n';
    if ((*north_west(nw))[0][0] != 2.0) throw "(*north_west(nw))[0][0] != 2.0";

    std::cout << "south_east of north_west:\n" << *south_east(nw) << '\n';
    if ((*south_east(nw))[0][0] != 4.0) throw "(*south_east(nw))[0][0] != 4.0";

    std::cout << "nw.first_address() == " << nw.first_address() 
	      << ", &(*nw)[0][0] == " << &(*nw)[0][0] << '\n';
    if (nw.first_address() != &(*nw)[0][0]) throw "Inconsistency in address calculation";
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
    test(dcc, "dense2D col-major");
    test(mdc, "pure Morton");
    test(mcc, "Hybrid col-major");

    return 0;
}
