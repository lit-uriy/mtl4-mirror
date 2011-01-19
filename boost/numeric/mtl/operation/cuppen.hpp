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
#include <boost/numeric/mtl/operation/iota.hpp>
#include <boost/numeric/mtl/operation/secular.hpp>
#include <boost/numeric/mtl/operation/sort.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/utility/domain.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>

namespace mtl { namespace matrix {

/// Eigenvalues of triangle matrix A with Cuppen's divide and conquer algorithm
/** Eigenvalues are returned in vector lambda. A is overwritten. **/
template <typename Matrix, typename Vector>
void inline cuppen_inplace(Matrix& A, Matrix& Q, Vector& lambda)
{
    using std::abs; using mtl::irange; using mtl::imax; using mtl::iall;

    typedef typename Collection<Matrix>::value_type     value_type;
    typedef typename Collection<Matrix>::size_type      size_type;
    typedef dense_vector<size_type>                     size_vector; // todo: with type trait

    size_type        nrows= num_rows(A);
    MTL_THROW_IF(nrows != num_cols(A), matrix_not_square());
    const value_type zero= 0, one= 1;   
    
    if (nrows == 1){
	lambda[0]= A[0][0];
	Q= one;
    } else {
	size_type     m= size_type(nrows/2);
	irange        till_m(m), from_m(m, imax);

	size_vector   perm(nrows);
	Matrix        T1(A[till_m][till_m]), T2(A[from_m][from_m]),                               // sub-matrices of A
	              Q0(nrows, nrows), Q1(Q0[till_m][till_m]), Q2(Q0[from_m][from_m]);           // Q0 and sub-matrices
	Vector        v(nrows, zero), diag(nrows), lambda1(diag[till_m]), lambda2(diag[from_m]);  // sub-vectors of diag

	//DIVIDE
	value_type    b= A[m-1][m];
	T1[m-1][m-1]-= abs(b);
	T2[0][0]-= abs(b);

	v[m-1]= b > zero ? one : -one;
	v[m]= one;

	cuppen_inplace(T1, Q1, lambda1);
	cuppen_inplace(T2, Q2, lambda2);
	Q0[till_m][from_m]= zero; Q0[from_m][till_m]= zero; // zero out non-diagonal blocks

	iota(perm);
	sort(diag, perm);

	// CONQUER, start with eq. (3.0.2) using rows (not columns)
	v[till_m]= b < zero ? Vector(-trans(Q1[m-1][iall])) : trans(Q1[m-1][iall]); 
	v[from_m]= trans(Q2[0][iall]);

	// permutation on v
	mtl::matrix::traits::permutation<>::type P= mtl::matrix::permutation(perm); 
	Vector v1(P * v);
	
	lambda= secular(v1, diag, abs(b));   // solve secular equation 

	//Lemma 3.0.2  ... calculate eigenvectors
	Matrix Q_tilde(nrows, nrows);
	for (size_type i = 0; i < nrows; i++) {
	    Vector    li(nrows, lambda[i]), lambda_i(ele_quot(v1, diag - li));
	    Q_tilde[iall][i]= lambda_i / two_norm(lambda_i); // normalized eigenvector in Matrix Q 
	}
	Q= Q0 * P * Q_tilde;
    }     
}

/// Eigenvalues of triangle matrix A with Cuppen's divide and conquer algorithm
/** Eigenvalues are returned in vector lambda. A is copied. **/
// A not as reference to force copy
template <typename Matrix, typename Vector>
void inline cuppen(Matrix A, Matrix& Q, Vector& lambda)
{
    cuppen_inplace(A, Q, lambda);
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_CUPPEN_INCLUDE
