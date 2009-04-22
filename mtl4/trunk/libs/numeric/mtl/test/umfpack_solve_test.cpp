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

// Example build call: scons -D check=1 with_blas=1 blas_ldflags=-lblas with_umfpack=1 umfpack_path=/u/pgottsch/Software/UMFPACK-5.3.0 amd_path=/u/pgottsch/Software/AMD ufconfig_path=/u/pgottsch/Software/UFconfig umfpack_solve_test

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;  

template <typename Matrix>
void test(const Matrix&, const char* name)
{
    using mtl::Collection;
    
    typedef typename Collection<Matrix>::value_type   value_type;
    
    value_type array[5][5]= {{ 2.,  3.,  0.,  0.,  0.},
			     { 3.,  0.,  4.,  0.,  6.},
			     { 0., -1., -3.,  2.,  0.},
			     { 0.,  0.,  1.,  0.,  0.},
			     { 0.,  4.,  2.,  0.,  1.}};
    Matrix A(array);
    crop(A);
    if (A.nnz() != 12) 
	throw "Matrix should have 12 non-zeros!";

    value_type                      b_array[5]= {8., 45., -3., 3., 19.};
    mtl::dense_vector<value_type>   x(5), b(b_array);

    cout << name << "\nA = \n" << A << "b = " << b << "\n";

    mtl::matrix::umfpack::solver<Matrix> solver(A);
    int status= solver(x, b);
    // int status= umfpack_solve(A, x, b); // creates solver on the fly
    cout << "A \\ b = " << x << "\n\n";

    for (int i= 0; i < 5; i++) 
	if (std::abs(x[i] - value_type(i+1)) > 0.01)
	    throw "Wrong result!";

    {
	mtl::matrix::inserter<Matrix> ins(A);
	ins[1][2] << 5.;
    }
    b[1]= 48.;
    cout << "\nA = \n" << A << "b = " << b << "\n";
    solver.update_numeric();

    status= solver(x, b);
    cout << "A \\ b = " << x << "\n\n";

    for (int i= 0; i < 5; i++) 
	if (std::abs(x[i] - value_type(i+1)) > 0.01)
	    throw "Wrong result after update_numeric!";

    {
	mtl::matrix::inserter<Matrix> ins(A);
	ins[3][4] << 2.;
    }
    b[3]= 13.;
    cout << "\nA = \n" << A << "b = " << b << "\n";
    solver.update();

    status= solver(x, b);
    cout << "A \\ b = " << x << "\n\n";

    for (int i= 0; i < 5; i++) 
	if (std::abs(x[i] - value_type(i+1)) > 0.01)
	    throw "Wrong result after update!";
}



int test_main(int argc, char* argv[])
{
#ifdef MTL_HAS_UMFPACK
    using namespace mtl;
    typedef matrix::parameters<col_major>           col_para;

    test(compressed2D<complex<double> >(),          "complex<double> row-major");
    test(compressed2D<complex<double>, col_para>(), "complex<double> column-major");

    test(compressed2D<complex<float> >(),           "complex<float> row-major");
    test(compressed2D<complex<float>, col_para>(),  "complex<float> column-major");

    test(compressed2D<double>(),                    "double row-major");
    test(compressed2D<double, col_para>(),          "double column-major");

    test(compressed2D<float>(),                     "float row-major");
    test(compressed2D<float, col_para>(),           "float column-major");

#else
    std::cout << "Test is ignored when MTL_HAS_UMFPACK is not defined\n";
#endif

    return 0;
}
