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
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

using namespace mtl;
using namespace std;  

typedef complex<double> ct;

double value(double)
{
    return 7.0;
}

complex<double> value(complex<double>)
{
    return ct(7.0, 1.0);
}

// scaled value
double svalue(double)
{
    return 14.0;
}

ct svalue(ct)
{
    return ct(14.0, 2.0);
}

// conjugated value
double cvalue(double)
{
    return 7.0;
}

ct cvalue(ct)
{
    return ct(7.0, -1.0);
}

// complex scaled value
ct csvalue(double)
{
    return ct(0.0, 7.0);
}

ct csvalue(ct)
{
    return ct(-1.0, 7.0);
}


template <typename Matrix>
void test(Matrix& matrix, const char* name)
{
    typedef typename Collection<Matrix>::value_type value_type;
    value_type  ar[][3] = {{3., 9., 0.},
			   {1., 2., 7.},
			   {9., 6., 8.}};
    const Matrix A(ar);
    
    // trans(A)[0][1]= 11.0;
    
    cout << "trans(A)[0][1]= " << trans(A)[0][1] << "\n";

    if (trans(A)[0][1] != value_type(1.))
	throw "constant transposing wrong";
}


template <typename Matrix>
void mutable_test(Matrix& matrix, const char* name)
{
    typedef typename Collection<Matrix>::value_type value_type;
    value_type  ar[][3] = {{3., 9., 0.},
			   {1., 2., 7.},
			   {9., 6., 8.}};
    Matrix A(ar);
    
    trans(A)[0][1]= 11.0;
    
    cout << "trans(A)[0][1]= " << trans(A)[0][1] << "\n";

    if (trans(A)[0][1] != value_type(11.))
	throw "transposing wrong";
}


int test_main(int argc, char* argv[])
{
    unsigned size= 3; 
    if (argc > 1) size= atoi(argv[1]); 

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppled_2_row_mask>  d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);


    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(mzd, "Morton Z-order");
    test(d2r, "Hybrid 2 row-major");
    test(cr, "Compressed row major");
    test(cc, "Compressed column major");
    test(drc, "Dense row major complex");
    test(crc, "Compressed row major complex");

    mutable_test(dr, "Dense row major");
    mutable_test(dc, "Dense column major");
    mutable_test(mzd, "Morton Z-order");
    mutable_test(d2r, "Hybrid 2 row-major");
    mutable_test(drc, "Dense row major complex");

    return 0;
}
