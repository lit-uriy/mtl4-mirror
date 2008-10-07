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
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/laplacian_setup.hpp> 
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/operation/print.hpp>


using namespace std;  


template <typename Matrix>
void test2(Matrix& matrix, unsigned dim1, unsigned dim2)
{
    Matrix tmp;
    laplacian_setup(tmp, dim1, dim2);
    swap(matrix, tmp);
}



template <typename Matrix>
void test(Matrix& matrix, unsigned dim1, unsigned dim2, const char* name)
{
    cout << "\n" << name << "\n";
    test2(matrix, dim1, dim2);
    cout << "Laplacian matrix:\n" << matrix << "\n";
    
    if (dim1 > 1 && dim2 > 1) {
	unsigned size= dim1 * dim2;
	if (num_rows(matrix) != size)
	    throw "wrong number of rows";
	if (num_cols(matrix) != size)
	    throw "wrong number of columns";

	typename mtl::Collection<Matrix>::value_type four(4.0), minus_one(-1.0), zero(0.0);
	if (matrix[0][0] != four)
	    throw "wrong diagonal";
	if (matrix[0][1] != minus_one)
	    throw "wrong east neighbor";
	if (matrix[0][dim2] != minus_one)
	    throw "wrong south neighbor";
	if (dim2 > 2 && matrix[0][2] != zero)
	    throw "wrong zero-element";
	if (matrix[1][0] != minus_one)
	    throw "wrong west neighbor";
	if (matrix[dim2][0] != minus_one)
	    throw "wrong north neighbor";
	if (dim2 > 2 && matrix[2][0] != zero)
	    throw "wrong zero-element";
    }
}


template <typename Vector>
void vtest2(Vector& vector, unsigned dim1, unsigned dim2)
{
    Vector tmp(dim1*dim2, 0.0);
    if (dim1 > 1 && dim2 > 1)
	tmp[2]= 4.0;
    swap(vector, tmp);
}



template <typename Vector>
void vtest(Vector& vector, unsigned dim1, unsigned dim2, const char* name)
{
    cout << "\n" << name << "\n";
    vtest2(vector, dim1, dim2);
    cout << "Vector after swap:\n" << vector << "\n";
    
    if (dim1 > 1 && dim2 > 1) {
	if (size(vector) != dim1*dim2)
	    throw "wrong number of elements";

	typename mtl::Collection<Vector>::value_type four(4.0), minus_one(-1.0), zero(0.0);
	if (vector[2] != four)
	    throw "wrong value in vector";
    }
}




int test_main(int argc, char* argv[])
{
    using namespace mtl;

    unsigned dim1= 3, dim2= 4;

    if (argc > 2) {
	dim1= atoi(argv[1]); 
	dim2= atoi(argv[2]);
    }
    unsigned size= dim1 * dim2; 

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppled_2_row_mask>  d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);

    test(dr, dim1, dim2, "Dense row major");
    test(dc, dim1, dim2, "Dense column major");
    test(mzd, dim1, dim2, "Morton Z-order");
    test(d2r, dim1, dim2, "Hybrid 2 row-major");
    test(cr, dim1, dim2, "Compressed row major");
    test(cc, dim1, dim2, "Compressed column major");
    test(drc, dim1, dim2, "Dense row major complex");
    test(crc, dim1, dim2, "Compressed row major complex");

    dense_vector<float>                                       cv(size);
    dense_vector<float, mtl::vector::parameters<row_major> >  rv(size);

    vtest(rv, dim1, dim2, "Dense row vector");
    vtest(cv, dim1, dim2, "Dense column vector");

    return 0;
}
