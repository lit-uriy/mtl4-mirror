
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


using namespace std;


double f(double i, double size) 
{ 
    return exp(-10*(i/size)*(i/size)); 
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    unsigned size= 4, n= size*size;

    dense_vector<double>					 c(size*size);
    dense_vector<double>                    		 b(size*size), x(size*size),  b2(size*size), b3(size*size), r(size*size);
    //dense2D<double>                                      dr(size, size), Q(size, size), R(size, size);
    dense2D<double, matrix::parameters<col_major> >      A(n, n), P(16, 8), R(8, 16), B(n,n), C(n,n);;

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
	//std::cout<< "A=\n" << A << "\n";
	//for (int i = 0; i < size*size; i++) {
	//	b[i]=f(i, size*size);
	//}
	//std::cout<< "b=" << b << "\n";
	b=1.0;

	//amg_setup(A, x, b, 4);
	boost::tie(A, B, C)= amg_setup(A, x, b, 4);
	std::cout<< "A=\n" << A << "\n";
//	std::cout<< "B=\n" << B << "\n";
//	std::cout<< "C=\n" << C << "\n";

//	itl::pc::diagonal<dense2D<double, matrix::parameters<col_major> > >     Ident(A);
//	itl::cyclic_iteration<double> iter(b, 500, 1.e-6, 0.0, 5);
//	bicgstab(A, x, b, Ident, iter);
//	std::cout<< "x=" << x << "\n";
// 	c= amg_coarsepoints_default(A,b);
// 	std::cout<< "C_default=" << c << "\n";
// 	c= amg_coarsepoints_simple(A,b);
// 	std::cout<< "C_simple= " << c << "\n";
// 	c= amg_coarsepoints_notay(A,b, 0.25);
// 	std::cout<< "C_notay=  " << c << "\n";
// 	//c= amg_coarsepoints_rs(A,b, 0.25);
// 	//std::cout<< "C_rs=     " << c << "\n";
// 
// 	P=amg_prolongate(A,c,0.25);
// 	std::cout<< "P=    \n" << P << "\n";
// 	
// 	R=amg_restict_simple(A,c,0.25);
// 	//std::cout<< "C_simple= " << c << "\n";
// 	std::cout<< "GROB=    \n" << R*A*trans(R) << "\n";
// 
// 	R=amg_restict_average(A,c,0.25);
// 	P=0.5*trans(R);
// 	R=0.25*R;
// 	std::cout<< "C= " << c << "\n";
// 	std::cout<< "GROB1=    \n" << A*trans(R) << "\n";
// 	std::cout<< "R*A*P=    \n" << R*A*P << "\n";
// 
// 	//relaxed_jacobi(A,x,b,1000,0.25);
//	gauss_seidel(A,x,b,1000,0.00025);
//	r=A*x-b;
//	std::cout<< "Two_norm(r)=" << two_norm(r) << "\n"; 
// 	//amg_multigrid(A,x,b,2,10);
// 	//c= amg_multigrid_setup(A,x, 2, 10);

    return 0;
}

