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

#ifndef MTL_MATRIX_CUPPEN_INCLUDE
#define MTL_MATRIX_CUPPEN_INCLUDE

#include <cmath>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/diagonal.hpp>
#include <boost/numeric/mtl/operation/iota.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/sequal.hpp>
#include <boost/numeric/mtl/operation/sort.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/utility/domain.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>


namespace mtl { namespace matrix {


/// Eigenvalues of triangel matrix A with cuppens divide and conquer algorithm
// Return Diagonalmatrix with eigenvalues as diag(A)
template <typename Matrix, typename Vector>
void inline cuppen(const Matrix& A, Matrix& Q, Matrix& L, Vector& p)
{
    using std::abs; using mtl::signum; using mtl::real;
    using mtl::irange; using mtl::imax; using mtl::iall;

    typedef typename Collection<Matrix>::value_type     value_type;
    typedef typename Collection<Matrix>::size_type      size_type;
    typedef typename Collection<Vector>::value_type     vec_value_type;
    typedef typename mtl::traits::domain<Matrix>::type  vec_type;

    size_type        ncols = num_cols(A), nrows = num_rows(A), m, n;
    value_type       zero= 0, one= 1;
    vec_value_type   zerovec= 0;
    Matrix           T(nrows,ncols), Q0(Q);
    
    MTL_THROW_IF(ncols != nrows, matrix_not_square());
    vec_type   v(nrows, zero), v1(nrows, zero), v2(nrows), diag(nrows, zero), lambda(nrows, zero);
    Vector     perm(nrows, zerovec), permdiag(nrows, zerovec);;
    
    if (ncols == 1){
	L= A;
	Q= 1;
    } else {
	m= size_type(nrows/2);
	n= nrows - m;
	Vector   perm1(m), perm2(n), perm_intern(nrows, zerovec);
	Matrix   T1(m, m), T2(n, n), Q1(m, m), Q2(n, n), L1(m, m), L2(n, n);

	//DIVIDE
	value_type b(A[m-1][m]);
	irange till_m(m), from_m(m, imax);

	T1= A[till_m][till_m]; 
	T1[m-1][m-1]-= abs(b);

	T2= A[from_m][from_m]; 
	T2[0][0]-= abs(b);

	v[m-1]= b > zero ? one : -one;
	v[m]= one;

	iota(perm1); iota(perm2);

	cuppen(T1, Q1, L1, perm1);
	cuppen(T2, Q2, L2, perm2);

	// permutation in global notation
	for (size_type i = 0; i < n; i++)
	    perm2[i]+= m;

	perm_intern[till_m]= perm1;
	perm_intern[from_m]= perm2;
      
	L[till_m][till_m]= L1;  L[till_m][from_m]= 0;
	L[from_m][till_m]= 0;   L[from_m][from_m]= L2;
	
	T[till_m][till_m]= T1;  T[till_m][from_m]= 0;
	T[from_m][till_m]= 0;   T[from_m][from_m]= T2;

	diag= diagonal(L);

	iota(perm);
	sort(diag, perm);

	// CONQUER
	v[till_m]= trans(Q1[m-1][iall]);
	if (b < zero)
	    v[till_m]*= -one;
	v[from_m]= trans(Q2[0][iall]);

#if 0	  
	std::cout << "Q1 is\n" << Q1;
	std::cout << "Q1*trans(Q1) is\n" << Matrix(Q1*trans(Q1));
	std::cout << "Q1*L*trans(Q1) is\n" << Q1*L1*trans(Q1);
	std::cout << "v_1 is\n" << v[till_m] << '\n';

	std::cout << "Q2 is\n" << Q2;
	std::cout << "Q2*trans(Q2) is\n" << Matrix(Q2*trans(Q2));
	std::cout << "Q2*L2*trans(Q2) is\n" << Q2*L2*trans(Q2);
	std::cout << "v_2 is\n" << v[from_m] << '\n';
#endif
	// permutation on v
	v1= permutation(perm) * v;

	// solve secular equation 
	lambda= secular(lambda, v1, diag, abs(b));

	//Lemma 3.0.2  ... calculate eigenvectors
	for (size_type i = 0; i < nrows; i++) {
	    vec_type    lambda_i(nrows, lambda[i]), test(diag - lambda_i);

	    // lambda_i= ele_quod(v, test);
	    for(size_type k = 0; k < nrows; k++)
		test[k]=1/test[k];
	    lambda_i= ele_prod(test, v1);

	    Q[iall][i]= lambda_i / two_norm(lambda_i); // normalized eigenvector in Matrix Q
	}
	
	L= mtl::vector::diagonal(lambda); // diagonal matrix with eigenvalues
        Matrix Q01(Q);
	Q0[till_m][till_m]= Q1;  Q0[till_m][from_m]= 0;
	Q0[from_m][till_m]= 0;   Q0[from_m][from_m]= Q2;
	Q=Q0* permutation(perm)*Q01;
	std::cout << "Q is\n" << Q;
	std::cout << "Q*L*trans(Q) is\n" << Q*L*trans(Q);
    }  
    p= perm;
   
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_CUPPEN_INCLUDE

