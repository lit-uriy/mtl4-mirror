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


int test_main(int argc, char* argv[])
{
    // For a more realistic example set sz to 1000 or larger
    const int N = 3;

    typedef mtl::compressed2D<double>  matrix_type;
    typedef mtl::dense_vector<double>  vector_type;
    mtl::compressed2D<double>          A(N, N);
    setup(A);
       
    itl::pc::ilu_0<matrix_type>        P(A);
    
    if(N < 11)
	dense_ilu_0(A, P.get_L(), P.get_U());

    mtl::dense_vector<double> x(N, 3.0), x2(N), Px(N), x3(N), x4(N), x5(N);

    matrix_type L(P.get_L()), U(P.get_U()), UT(trans(U));

    std::cout << "L is\n" << L << "U is \n" << U;

    x2= strict_upper(U) * x;
    for (int i= 0; i < N; i++)
	x2[i]+= 1. / U[i][i] * x[i];
    std::cout << "U*x = " << x2 << "\n";

    Px= L * x2 + x2;
    std::cout << "P*x = (L+I)*U*x = " << Px << "\n";

    x4= unit_lower_trisolve(L, Px);
    std::cout << "L^{-1} * Px = " << x4 << "\n";

    if (two_norm(vector_type(x4 - x2)) > 0.01) throw "Error in unit_lower_trisolve.";

    x5= inverse_upper_trisolve(U, x4);
    std::cout << "U^{-1} * L^{-1} * Px = " << x5 << "\n";

    if (two_norm(vector_type(x5 - x)) > 0.01) throw "Error in inverse_upper_trisolve.";

    x3= solve(P, Px);
    std::cout << "solve(P, Px) = " << x3 << "\n";
    if (two_norm(vector_type(x3 - x)) > 0.01) throw "Error in solve.";


    // Now test adjoint solve
    x2= trans(L) * x + x;
    std::cout << "\n\nNow test adjoint solve\n(L+I)^T*x = " << x2 << "\n";

    //Px= trans(strict_upper(U)) * x2;
    Px= strict_lower(UT) * x2;
    for (int i= 0; i < N; i++)
	Px[i]+= 1. / U[i][i] * x2[i];
    std::cout << "P^T*x = ((L+I)*U)^T*x = " << Px << "\n";

    x4= inverse_lower_trisolve(adjoint(U), Px);
    std::cout << "U^{-T} * Px = " << x4 << "\n";

    if (two_norm(vector_type(x4 - x2)) > 0.01) throw "Error in inverse_lower_trisolve.";

    x5= unit_upper_trisolve(adjoint(L), x4);
    std::cout << "L^{-T} * U^{-T} * Px = " << x5 << "\n";
    if (two_norm(vector_type(x5 - x)) > 0.01) throw "Error in unit_upper_trisolve.";

    x3= adjoint_solve(P, Px);
    std::cout << "adjoint_solve(P, Px) = " << x3 << "\n";
    if (two_norm(vector_type(x3 - x)) > 0.01) throw "Error in adjoint_solve.";

    return 0;
}
