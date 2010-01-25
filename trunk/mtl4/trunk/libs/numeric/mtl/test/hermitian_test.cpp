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

//#include <boost/numeric/mtl/matrix/hermitian_view.hpp>

using namespace std;


template <typename Matrix>
void test(const Matrix&, const char* name)
{
    cout << name << " ... " << typeid(typename mtl::OrientedCollection<mtl::matrix::hermitian_view<Matrix> >::orientation).name() << '\n';
#if 1
    cout << name << '\n';
    Matrix A(3, 3), B(3, 3), C(3, 3);
    // Stupid test with real values --> use complex
    A= 2, 3, 4,
       1, 2, 3,
       3, 4, 5;
    B= 3, 4, 5,
       2, 3, 4,
       1, 2, 3;

    C= hermitian(B);
    cout << "hermitian(B) is\n" << C;

    C= A * hermitian(B);
    cout << "A * hermitian(B) is\n" << C;
#endif
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
    test(cr, "Compressed row major");
    test(drc, "Dense row major complex");

    // For better readability I don't want finish with a complex
    test(cc, "Compressed column major");

    return 0;
}
