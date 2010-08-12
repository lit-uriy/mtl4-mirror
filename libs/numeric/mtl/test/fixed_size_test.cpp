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
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

 
template <typename Matrix, typename Vector>
void test(const char* A_string, const char* v_string, const Matrix& A, const Vector&x)
{
    std::cout << "\n" << A_string << "ly sized matrix and " << v_string << "ly sized vector\nA is\n" << A;

    Matrix B(A + A);
    std::cout << "A+A = \n" << B;
    if (B[0][0] != 4.0) throw "wrong result in matrix addition.";

    B= A * A;
    std::cout << "A*A = \n" << B;
    if (B[0][0] != 16.0) throw "wrong result in matrix product.";

    Vector w(x + x);
    std::cout << "x = " << x << "\nw = x+x = " << w << "\n";
    if (w[0] != 6.0) throw "wrong result in vector addition.";

    w= A * x;
    std::cout << "A*x = " << w << "\n";
    if (w[0] != 18.0) throw "wrong result in matrix vector product.";
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    typedef vector::parameters<tag::col_major, vector::fixed::dimension<2>, true> fvec_para;
    typedef matrix::parameters<tag::row_major, mtl::index::c_index, mtl::fixed::dimensions<2, 2>, true> fmat_para;

    float ma[2][2]= {{2., 3.}, {4., 5.}}, va[2]= {3., 4.};
    
    dense2D<float>                   A_dyn(ma);
    dense2D<float, fmat_para>        A_stat(ma);
    dense_vector<float>              v_dyn(va);
    dense_vector<float, fvec_para>   v_stat(va);

    test("dynamic", "dynamic", A_dyn, v_dyn);
    test("dynamic", "static", A_dyn, v_stat);
    test("static", "dynamic", A_stat, v_dyn);
    test("static", "staticic", A_stat, v_stat);

    return 0;
}

