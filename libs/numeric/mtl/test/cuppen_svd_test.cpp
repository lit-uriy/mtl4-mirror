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

#include <cstdlib>
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
    if (size(v1) < 17) 
	cout << "A*v is     " << v1 << "\nalpha*v is " << v2 << '\n';
    if (two_norm(diff) > tol) throw "wrong eigenvector";
}

int test_main(int argc, char** argv)
{
    using namespace mtl;
    int size= 16;
    dense_vector<double>        eig, lambda(4),  lambda_b(size), eig_b(size);

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

    dense2D<double> B(size,size), BQ(size,size);
    B= 0; BQ= 0;
    
    for(int i= 1; i < size ; i++){
      B[i][i]= 4*i+6;
      B[i][i-1]= 1;
      B[i-1][i]= 1;
    }
    B[0][0]= 4;
//     std::cout << "B=" << B << "\n";

     eig_b= eigenvalue_symmetric(B,22);
    sort(eig_b);
    std::cout<<"eigenvalues  ="<< eig_b <<"\n";
    
    cuppen(B, BQ, lambda_b);
    std::cout<<"B  =\n"<< B <<"\n";
    std::cout<<"Q  =\n"<< BQ <<"\n";
    std::cout<<"eigenvalues  ="<< lambda_b <<"\n";
    
    eig_b-= lambda_b;
    std::cout<<"two_norm(diff)  ="<< two_norm(eig_b) <<"\n";
    if (two_norm(eig_b) > tol) throw "Cuppen computes wrong eigenvalues";

    for (unsigned i= 0; i < num_rows(B); i++)
	test_vector(B, lambda_b[i], dense_vector<double>(BQ[iall][i]));

    // Poisson equation cannot be solved yet, double eigenvalues are now correctly handled by the secular equation
    // but Q_tilde in cuppen contains nans (0/0)
    int lsize= 4;
    if (argc > 1) lsize= atoi(argv[1]);

#if 0
    dense2D<double> C(lsize, lsize), CQ(lsize, lsize);
    C= 0; CQ= 0;
    
    C[0][0]= 2;
    for(int i= 1; i < lsize; i++) {
	C[i][i]= 2;
	C[i][i-1]= -1;
	C[i-1][i]= -1;
    }
    cout << "The matrix of the 1D-Poisson equations I\n" << C << '\n';
	

    dense_vector<double> lambda_c(lsize);
    cuppen(C, CQ, lambda_c);

    if (lsize <= 100)
	cout << "The eigenvalues of the 1D-Poisson equations are " << lambda_c << '\n';
    if (lsize <= 20)
	cout << "The eigenvectors of the 1D-Poisson equations are\n" << CQ << '\n';

    for (unsigned i= 0; i < num_rows(C); i++)
	test_vector(C, lambda_c[i], dense_vector<double>(CQ[iall][i]));
#endif
    
    return 0;
}



