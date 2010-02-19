
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
#include <boost/utility.hpp>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <string>

using namespace std;


double f(double i, double size) 
{ 
    return exp(-10*(i/size)*(i/size)); 
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    unsigned size= 10, n= size*size;
    
    dense_vector<double>					 c(size*size);
    dense_vector<double>                    		b(size*size), x(size*size),  b2(size*size), b3(size*size), r(size*size);
//    dense2D<double>                                     A(n, n), dr(size, size), Q(size, size), R(size, size);
    dense2D<double, matrix::parameters<row_major> >      A(n, n), R(8, 16), B(n,n), C(n,n);;
    //compressed2D<double>      				A(n, n), R(8, 16), B(n,n), C(n,n);;
std::string s("blabla");
	std::cout<< s << "\n";

	//setup problem
	//laplacian_setup(A,size,size);
	for (int i=1; i< n-1; i++ ) {
		A[i][i]=2;
		A[i-1][i]=-1;
		A[i][i-1]=-1;
	}
	A[0][0]=2;
	A[n-1][n-1]=2;
	A[n-2][n-1]=-1;
	A[n-1][n-2]=-1;


	x=0; c=1;
	b=1.0;
	itl::cyclic_iteration<double> Iter(b, 1600, 1.e-3, 0.0, 5);
	int ny_pre(10), ny_post(10), ny(1);
	double omega(0.25), coarse_beta(0.125);

#if 0
	mg_hierarchy mgh(A, x, b, 4, rs, transf, avgr, coarse_beta);
	gs_hierarchy gsh(mgh);
	jacobi_hierarchy jsh(mgh, omega);
	mg_iteration(mgh, gsh, ny_pre, gsh, ny_post);

#endif
	multigrid_algo(A, x, b, Iter, 4, "notay", "trans", "avg", 2, ny_pre, ny_post, "gauss_seidel", "gauss_seidel", omega, coarse_beta, "inv");
  
	if (size < 11)
		std::cout<< "x=" << x << "\n";
	r=b-A*x;
	std::cout<< "two_norm=" << two_norm(r) << std::endl;
//	itl::pc::diagonal<dense2D<double, matrix::parameters<col_major> > >     P(A);
//	x=0;
//	bicgstab_2(A, x, b, P, Iter);
//	if (size < 11)
//		std::cout<< "x=" << x << "\n";
	
    return 0;
}

