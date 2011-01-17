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

// With contributions from Cornelius Steinhardt

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/operation/cuppen.hpp>

using namespace std;

const double 			tol= 1.0e-5;

template <typename Matrix, typename Value, typename Vector>
void test_vector(const Matrix& A, const Value& alpha, const Vector& v)
{
    Vector v1(A*v), v2(alpha*v), diff(v1-v2);
    cout << "A*v is     " << v1 << "\nalpha*v is " << v2 << '\n';
    if (two_norm(diff) > tol) throw "wrong eigenvector";
}

int test_main(int , char**)
{
    using namespace mtl;

    dense_vector<double>        eig, lambda(4);

    double array[][4]= {{1,  2,   0,  0},
                        {2, -9,  -2,  0},
                        {0, -2,   1,  3},
                        {0,  0,   3, 10}};
    dense2D<double> A(array), Q(4,4);
    std::cout << "A=\n" << A << "\n";

    eig= eigenvalue_symmetric(A,22);
    sort(eig);
    std::cout<<"eigenvalues  ="<< eig <<"\n";
    
    cuppen(A, Q, lambda);
    std::cout<<"A  =\n"<< A <<"\n";
    std::cout<<"Q  =\n"<< Q <<"\n";
    std::cout<<"eigenvalues  ="<< lambda <<"\n";
    
    eig-= lambda;
    std::cout<<"two_norm(diff)  ="<< two_norm(eig) <<"\n";
    if (two_norm(eig) > tol) throw "Cuppen computes wrong eigenvalues";

    for (unsigned i= 0; i < num_rows(A); i++)
	test_vector(A, lambda[i], dense_vector<double>(Q[iall][i]));

    return 0;
}



