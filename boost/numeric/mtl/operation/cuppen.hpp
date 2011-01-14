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


/// Eigenvalues of triangle matrix A with Cuppen's divide and conquer algorithm
/** Eigenvalues are returned in vector lambda. A is overwritten. **/
template <typename Matrix, typename Vector>
void inline cuppen(Matrix& A, Matrix& Q, Vector& lambda)
{
    using std::abs; using mtl::signum; using mtl::real;
    using mtl::irange; using mtl::imax; using mtl::iall;

    typedef typename Collection<Matrix>::value_type     value_type;
    typedef typename Collection<Matrix>::size_type      size_type;
    typedef dense_vector<size_type>                     size_vector; // todo: with type trait

    size_type        nrows= num_rows(A);
    value_type       zero= 0, one= 1;
    Matrix           Q0(Q);
    
    MTL_THROW_IF(nrows != num_cols(A), matrix_not_square());
    Vector          v(nrows, zero), v1(nrows, zero), diag(nrows, zero);
    size_vector     perm(nrows), permdiag(nrows);
    
    if (nrows == 1){
	lambda[0]= A[0][0];
	Q= 1;
    } else {
	size_type     m= size_type(nrows/2), n= nrows - m;
	irange        till_m(m), from_m(m, imax);

	size_vector   perm1(m), perm2(n);
	Matrix        T1(A[till_m][till_m]), T2(A[from_m][from_m]), Q1(m, m), Q2(n, n); // sub-matrices
	Vector        lambda1(m), lambda2(n);

	//DIVIDE
	value_type    b(A[m-1][m]);
	T1[m-1][m-1]-= abs(b);
	T2[0][0]-= abs(b);

	v[m-1]= b > zero ? one : -one;
	v[m]= one;

	cuppen(T1, Q1, lambda1);
	cuppen(T2, Q2, lambda2);

	diag[till_m]= lambda1;
	diag[from_m]= lambda2;
	
	iota(perm);
	sort(diag, perm);

	// CONQUER (3.0.2) with rows (not columns)
	v[till_m]= b < zero ? Vector(-trans(Q1[m-1][iall])) : trans(Q1[m-1][iall]); 
	v[from_m]= trans(Q2[0][iall]);

	// permutation on v
	mtl::matrix::traits::permutation<>::type P= mtl::matrix::permutation(perm); 
	v1= P * v;

	// solve secular equation 
	lambda= secular(v1, diag, abs(b));

	//Lemma 3.0.2  ... calculate eigenvectors
	Matrix Q_tilde(nrows, nrows);
	for (size_type i = 0; i < nrows; i++) {
	    Vector    li(nrows, lambda[i]), lambda_i(ele_quot(v1, diag - li));
	    Q_tilde[iall][i]= lambda_i / two_norm(lambda_i); // normalized eigenvector in Matrix Q
	}
	Q0[till_m][till_m]= Q1;  Q0[till_m][from_m]= 0;
	Q0[from_m][till_m]= 0;   Q0[from_m][from_m]= Q2;

	Q= Q0 * P * Q_tilde;
    }     
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_CUPPEN_INCLUDE

