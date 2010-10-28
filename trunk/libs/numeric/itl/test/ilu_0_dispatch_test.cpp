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

#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

template <typename Matrix>
void setup(Matrix& A)
{
    const int n= num_rows(A);
    A= 1.0;
    mtl::matrix::inserter<Matrix, mtl::update_plus<double> > ins(A);

    for (int i= 0; i < 2 * n; i++) {
	int r= rand()%n, c= rand()%n;
	ins[r][c] << -1;
	ins[r][r] << 1;
    }
}


template <typename At, typename Lt, typename Ut>
void dense_ilu_0(const At& As, const Lt& Ls, const Ut& Us)
{
    mtl::dense2D<double> LU(As);
     
    const int n= num_rows(LU);
    for (int i= 1; i < n; i++) 
	for (int k= 0; k < i; k++) {
	    LU[i][k]/= LU[k][k];
	    for (int j= k + 1; j < n; j++)
		if (LU[i][j] != 0)
		    LU[i][j]-= LU[i][k] * LU[k][j];
	}
    std::cout << "Factorizing A = \n" << As << "-> LU = \n" << LU;
    // std::cout << "L = \n" << Ls << "\nU = \n" << Us;

    if (std::abs(LU[1][2] - Ls[1][2]) > 0.001) 
	throw "Wrong value in L for sparse ILU(0) factorization";

    if (std::abs(LU[2][2] - 1. / Us[2][2]) > 0.001)
	throw "Wrong value in U for sparse ILU(0) factorization";
}


int test_main(int, char**)
{
    // For a more realistic example set sz to 1000 or larger
    const int N = 5;

    typedef mtl::dense2D<double>       matrix_type;
    typedef mtl::dense_vector<double>  vector_type;
    matrix_type                        A(N, N);
    setup(A);
       
    itl::pc::ilu_0<matrix_type>        P(A);
    
    mtl::dense_vector<double> x(N), x2(N), Px(N), x3(N), x4(N), x5(N);

    for (unsigned i= 0; i < num_rows(x); i++)
	x[i]= i+1;

    std::cout << "A is\n" << A;
    x2= A * x;
    std::cout << "x2= A * x = " << x2 << "\n";

    x3= solve(P, x2);
    std::cout << "solve(P, x2) = " << x3 << " (should be [1,2,..,N])\n";
    if (two_norm(vector_type(x - x3)) > 0.00001)
	throw "Wrong result";

    // Now test adjoint solve
    x4= trans(A) * x;
    std::cout << "x4= adjoint(A) * x = " << x4 << "\n";

    x5= adjoint_solve(P, x4);
    std::cout << "adjoint_solve(P, x4) = " << x5 << " (should be [1,2,..,N])\n";
    if (two_norm(vector_type(x - x5)) > 0.00001)
	throw "Wrong result";

    return 0;
}
