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
    cout << "\n" << name << "\n";

    typedef typename mtl::Collection<Matrix>::value_type  Scalar;
    typedef typename mtl::dense_vector<Scalar>            Vector;

    unsigned size= num_cols(A);
    Matrix L(size, size), U(size, size);

    for (unsigned i= 0; i < size; i++)
	for(unsigned j= 0; j < size; j++) {
	    U[i][j]= i <= j ? Scalar(i+j+2) : Scalar(0);
	    L[i][j]= i > j ? Scalar(i+j+1) : (i == j ? Scalar(1) : Scalar(0));
	}
    
    cout << "L is:\n" << L << "U is:\n" << U;
    A= L * U;

    Vector v(size);
    for (unsigned i= 0; i < size; i++)
	v[i]= Scalar(i);

    Vector w( A*v );

    cout << "A is:\n" << A;

    Matrix LU(A);
    lu(LU);
    cout << "LU decomposition of A is:\n" << LU;

    Matrix I(size, size);
    I= Scalar(1);
    Matrix tmp(I + strict_lower(LU)), A2(tmp * upper(LU));
    cout << "L * U is:\n" << A2;

    Matrix B( lu_f(A) );

    Vector v2( upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), w)) );

    cout << "LU decomposition of A (as function result) is:\n" << B;
    cout << "upper(LU) is:\n" << upper(LU) << "strict_lower(LU) is:\n" << strict_lower(LU);
    cout << "v2 is " << v2 << "\n";

    if (abs(v[1] - v2[1]) > 0.1) throw "Error using tri_solve";

    Vector v3( lu_solve(A, w) );
    if (abs(v[1] - v3[1]) > 0.1) throw "Error in solve";
}



int test_main(int argc, char* argv[])
{
    using namespace mtl;
    unsigned size= 5;
    
#if 0
    // Might be added for sparse matrices later
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);
    compressed2D<complex<double> >                       cz(size, size);
#endif

    dense2D<double>                                      dr(size, size);
    dense2D<complex<double> >                            dz(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);

#if 0
    test(cr, "Row-major sparse");
    test(cc, "Column-major sparse");
    test(cz, "Row-major complex sparse");
#endif

    test(dr, "Row-major dense");
    test(dr, "Row-major dense with complex numbers");
    test(dc, "Column-major dense");

    return 0;
}
