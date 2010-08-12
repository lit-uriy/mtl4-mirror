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

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

template <typename At, typename Ut>
void dense_ic_0(const At& As, const Ut& Us)
{
    mtl::dense2D<double> U(upper(As));
     
    const int n= num_rows(U);

    for (int k= 0; k < n; k++) {
	double dia= U[k][k]= sqrt(U[k][k]);
	for (int i = k + 1; i < n; i++) {
	    double d= U[k][i] /= dia;
	    for (int j = k + 1; j <= i; j++)
		if (U[j][i] != 0.0)
		    U[j][i] -= d * U[k][j];
	}
    } 

    std::cout << "Factorizing A = \n" << As << "-> U = \n" << with_format(U, 6, 2)
	      << "trans(U) * U = \n" << with_format(mtl::dense2D<double>(trans(U) * U), 6, 2);

    if (std::abs(U[2][3] - Us[2][3]) > 0.001) 
	throw "Wrong value in L for sparse IC(0) factorization";

    if (std::abs(U[3][3] - 1. / Us[3][3]) > 0.001)
	throw "Wrong value in U for sparse IC(0) factorization";
}


int main()
{
    // For a more realistic example set sz to 1000 or larger
    const int size = 3, N = size * size; 

    typedef mtl::compressed2D<double>  matrix_type;
    mtl::compressed2D<double>          A(N, N), dia(N, N);
    laplacian_setup(A, size, size);
    // dia= 1.0; A+= dia;
    
   
    itl::pc::ic_0<matrix_type>         P(A);
    mtl::dense_vector<double>          x(N, 1.0), b(N);
    
    if(size > 1 && size < 4)
	dense_ic_0(A, P.get_U());

    b = A * x;
    x= 0;
    
    itl::cyclic_iteration<double> iter(b, N, 1.e-6, 0.0, 5);
    cg(A, x, b, P, iter);
    
    return 0;
}
