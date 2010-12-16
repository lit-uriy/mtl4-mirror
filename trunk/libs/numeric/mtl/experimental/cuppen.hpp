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
#include <boost/utility.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/diagonal.hpp>
#include <boost/numeric/mtl/operation/givens.hpp>
#include <boost/numeric/mtl/operation/householder.hpp>
#include <boost/numeric/mtl/operation/qr.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>


namespace mtl { namespace matrix {


/// Eigenvalues of triangel matrix A with cuppens divide and conquer algorithm
// Return Diagonalmatrix with eigenvalues as diag(A)
template <typename Matrix>
void inline cuppen(const Matrix& A, Matrix& Q, Matrix& L)
{
    using std::abs; using mtl::signum; using mtl::real;
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Magnitude<value_type>::type      magnitude_type; // to multiply with 2 not 2+0i
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type        ncols = num_cols(A), nrows = num_rows(A), m, n;
    value_type       zero= math::zero(A[0][0]), one= math::one(A[0][0]);//, h00, h10, h11, beta, mu, a, b, tol;
  //  const magnitude_type two(2);
    Matrix           T(nrows,ncols);
    
    MTL_THROW_IF(ncols != nrows , matrix_not_square());

    if (ncols == 1){
      L[0][0]= A[0][0];
      Q[0][0]= 1;
    } else {
      m= size_type(nrows/2);
      n= nrows - m;
      Matrix           T1(m, m), T2(n, n), Q1(m,m),Q2(n,n),L1(m,m), L2(n,n);
      //DIVIDE
      value_type b(A[m-1][m]);
      T1= sub_matrix(A, 0, m, 0, m);
      T1[m-1][m-1]-= abs(b);
      std::cout<< "T1=" << T1 <<"\n";
      std::cout<< "n=" << n <<"\n";
      T2= sub_matrix(A, m, nrows, m, nrows);
      T2[0][0]-= abs(b);
      std::cout<< "T2=" << T2 <<"\n";
      
      std::cout<< "b=" << b <<"\n";
      dense_vector<value_type>    v(nrows, zero);
      if(b > zero)
	v[m-1]= one;
      else 
	v[m-1]= -one;      
      v[m]= one;
      std::cout<< "v=" << v <<"\n";
      cuppen(T1, Q1, L1);
      cuppen(T2, Q2, L2);
      
      
 
    }
    
   std::cout<< "L=" << L <<"\n";
std::cout<< "Q=" << Q <<"\n";

   
    
}

}} // namespace mtl::matrix


#endif // MTL_MATRIX_CUPPEN_INCLUDE

