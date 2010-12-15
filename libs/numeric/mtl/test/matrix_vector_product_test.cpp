// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
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
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/dense2D.hpp> 
#include <boost/numeric/mtl/matrix/laplacian_setup.hpp> 
#include <boost/numeric/mtl/vector/dense_vector.hpp> 
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/operation/operators.hpp>


using namespace std;  

template <typename MatrixA>
void test(MatrixA& A, unsigned dim1, unsigned dim2, const char* name)
{
    const unsigned max_print_size= 25;
    cout << "\n" << name << "\n";
    laplacian_setup(A, dim1, dim2);

    unsigned size= dim1 * dim2;
    mtl::dense_vector<double> v(size);
    for (unsigned i= 0; i < num_cols(A); i++)
	v[i]= A[12][i];

    // Resulting vector has same value type as matrix
    typedef typename mtl::Collection<MatrixA>::value_type rvalue_type;
    mtl::dense_vector<rvalue_type> w(size);

    w= A * v;
    //mult(A, v, w);

    if (size <= max_print_size)
	cout << "A= \n" << A << "\n\nv= " << v << "\n\nA*v= " << w << "\n";

    // Same test as in matrix product: resulting vector corresponds to column 12
    // Check for stencil below in the middle of the matrix
    //        1
    //     2 -8  2
    //  1 -8 20 -8  1
    //     2 -8  2
    //        1    
    if (dim1 == 5 && dim2 == 5) {
	rvalue_type twenty(20.0), two(2.0), one(1.0), zero(0.0), minus_eight(-8.0);
	if (w[12] != twenty) throw "wrong diagonal";
	if (w[13] != minus_eight) throw "wrong east neighbor";
	if (w[14] != one) throw "wrong east east neighbor";
	if (w[15] != zero) throw "wrong zero-element";
	if (w[17] != minus_eight) throw "wrong south neighbor";
	if (w[18] != two) throw "wrong south east neighbor";
	if (w[22] != one) throw "wrong south south neighbor";
    }

    w+= A * v;

    if (size <= max_print_size)
	cout << "w+= A*v= \n\n" << w << "\n";

    // Check for stencil, must be doubled now
    if (dim1 == 5 && dim2 == 5) {
	rvalue_type forty(40.0), four(4.0);
	if (w[12] != forty) throw "wrong diagonal";
	if (w[18] != four) throw "wrong south east neighbor";
    }

    w-= A * v;

    if (size <= max_print_size)
	cout << "w-= A*v= \n\n" << w << "\n";

    // Check for stencil, must be A*v now
    if (dim1 == 5 && dim2 == 5) {
	rvalue_type twenty(20.0), two(2.0);
	if (w[12] != twenty) throw "wrong diagonal";
	if (w[18] != two) throw "wrong south east neighbor";
    }
}



int test_main(int argc, char* argv[])
{
    using namespace mtl;

    unsigned dim1= 5, dim2= 5;

    if (argc > 2) {dim1= atoi(argv[1]); dim2= atoi(argv[2]);}
    unsigned size= dim1 * dim2; 

    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);

    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);

    test(cr, dim1, dim2, "Row-major sparse");
    test(cc, dim1, dim2, "Column-major sparse");

    test(dr, dim1, dim2, "Row-major dense");
    test(dc, dim1, dim2, "Column-major dense");

    return 0;
}
