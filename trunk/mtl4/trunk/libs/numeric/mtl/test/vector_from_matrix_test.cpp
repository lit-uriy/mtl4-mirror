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


using namespace std;  
    
    
#if 0

template <typename Matrix>
void test(Matrix& A, const char* name)
{
    typedef typename mtl::Collection<Vector>::value_type value_type;
    typedef mtl::dense_vector<float>                     Vector;

    std::cout << "\n" << name << "  --- A = \n" << A; 

    Vector vc(3), vr(3);
    vc= 2, 5, 8; vr= 1, 2, 3;

    cout << "A[iall][1] == " << A[iall][1] << "\n";
    if (norm(Vector(vc - A[iall][1])) > 0.1) throw "Wrong column vector";

    cout << "A[irange(1, imax)][1] == " << A[irange(1, imax)][1] << "\n";
    if (norm(Vector(vc - A[irange(1, imax)][1])) > 0.1) throw "Wrong column cub-vector";

    typename ColumnOfMatrix<Matrix>::type c(A[irange(1, imax)][1]);
    c[1]= 8.5;
    if (A[1][2] != 8.5) throw "Matrix modification (in column) did not work";
}
#endif
 

int test_main(int argc, char* argv[])
{
    using namespace mtl;
#if 0
    dense_vector<float> v(3);
    v= 2, 5, 8;
    test(v, "Reference");

    dense2D<float> A(3, 3);
    A= 1, 2, 3,
       4, 5, 6, 
       7, 8, 9;

    dense2D<float, matrix::parameters<col_major> > B(A);
    
    vector::strided_vector_ref<float> x(3, &A[0][1], 3);

    const dense2D<float> B(A);
    vector::strided_vector_ref<const float> xc(3, &B[0][1], 3);

    test(A, "Row-major matrix");
    test(B, "Column-major matrix");

    const dense2D<float>                                 C(A);
    const dense2D<float, matrix::parameters<col_major> > C(A);

    test(C, "Row-major matrix const");
    test(D, "Column-major matrix const");
#endif
    return 0;
}
 












