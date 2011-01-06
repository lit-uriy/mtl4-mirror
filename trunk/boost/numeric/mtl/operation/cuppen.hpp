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
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/sequal.hpp>
#include <boost/numeric/mtl/operation/sort.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>


namespace mtl { namespace matrix {


/// Eigenvalues of triangel matrix A with cuppens divide and conquer algorithm
// Return Diagonalmatrix with eigenvalues as diag(A)
template <typename Matrix>
void inline cuppen(const Matrix& A, Matrix& Q, Matrix& L)
{
    using std::abs; using mtl::signum; using mtl::real;
    using mtl::irange; using mtl::imax; using mtl::iall;

    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Magnitude<value_type>::type      magnitude_type; // to multiply with 2 not 2+0i
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type        ncols = num_cols(A), nrows = num_rows(A), m, n;
    value_type       zero= math::zero(A[0][0]), one= math::one(A[0][0]);//, h00, h10, h11, beta, mu, a, b, tol;
  //  const magnitude_type two(2);
    Matrix           T(nrows,ncols);
    
    MTL_THROW_IF(ncols != nrows , matrix_not_square());

     dense_vector<value_type>    v(nrows, zero), diag(nrows, zero), lambda(nrows, zero);
      dense_vector<size_type>     perm(nrows, zero);
    
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
      std::cout<< "n=" << n <<"\n";
      T2= sub_matrix(A, m, nrows, m, nrows);
      T2[0][0]-= abs(b);
      
//       std::cout<< "b=" << b <<"\n";
     
      if(b > zero)
	v[m-1]= one;
      else 
	v[m-1]= -one;      
      v[m]= one;
//       std::cout<< "v_=" << v <<"\n";
      cuppen(T1, Q1, L1);
      cuppen(T2, Q2, L2);
//       std::cout<< "v_2=" << v <<"\n";
 #if 0     
      std::cout<< "T1=" << T1 <<"\n";
      std::cout<< "T2=" << T2 <<"\n";
      std::cout<< "Q1=" << Q1 <<"\n";
      std::cout<< "Q2=" << Q2 <<"\n";
   
      std::cout<< "L1=" << L1 <<"\n";
      std::cout<< "L2=" << L2 <<"\n";
   #endif
      L[irange(0,m)][irange(0,m)]= L1;
      L[irange(m, imax)][irange(m, imax)]= L2;
     T= 0;
//       std::cout<< "L=" << L << "\n";
      T[irange(0,m)][irange(0,m)]= T1;
      T[irange(m, imax)][irange(m, imax)]= T2;
      diag= diagonal(L);
//       std::cout << "diag=" << diag << "\n";
      for (size_type i = 0; i < nrows; i++)
	perm[i]= i;
      //Diagonalmatrix sortieren
//       std::cout << "perm=" << perm << "\n";
//  std::cout<< "v_3=" << v <<"\n";
       
      sort(diag, perm);
//       std::cout << "diag=" << diag << "\n";
//       std::cout << "perm=" << perm << "\n";
//       std::cout<< "T=" << T <<"\n";
      //permutation on Matrix T
      mtl::matrix::traits::permutation<>::type P= mtl::matrix::permutation(perm);
//       std::cout << "\nP =\n" << P;    

      Matrix TP( P * A*P );
      std::cout << "\nTP =\n" << TP;   
      
      sort(perm, v); //permutation on v
//       std::cout << "perm =" << perm; 
//       std::cout << "v =" << v;
//       std::cout<< "Q1=" << Q1 <<"\n";
//       std::cout<< "Q2=" << Q2 <<"\n";
//       std::cout<< "m=" << m <<"\n";
      v[irange(0,m)]=Q1[irange(0,m)][m-1];
      v[irange(m,imax)]=Q2[irange(0,n)][0];
//       std::cout << "QQQ   v =" << v;
//       std::cout<< "diag= " << diag << "\n";
//        std::cout<< "abs(b)= " << abs(b) << "\n";
//       std::cout<< "hallo \n";
//       std::cout<<"roots  ="<< secular(lambda, v, diag, abs(b)) <<"\n";
      lambda= secular(lambda, v, diag, abs(b));
//       std::cout<< "lambda=" << lambda << "\n";
      //Lemma 3.0.2  ... calculate eigenvectors
      for(size_type i = 0; i < nrows; i++){
	  dense_vector<value_type>    test(nrows, zero), lambda_i(nrows, lambda[i]);
	  test=diag-lambda_i;
	  std::cout<< "test=" << test << "\n";
	  for(size_type k = 0; k < nrows; k++)
	    test[k]=1/test[k];
	  lambda_i= ele_prod(test, v);
	  std::cout<< "lambda_i=" << lambda_i << "\n";
	  test=TP*lambda_i;  //Permutation von Q beachten
	  std::cout<< "TESTing.......=" << test << "\n";
	  test=lambda[i]*lambda_i;  
	  std::cout<< "TESTing.......=" << test << "\n";
	  
	  
	  Q[irange(0, imax)][i]= lambda_i;
	 
      }
       L=mtl::vector::diagonal(lambda);
 
  
     
    }
    

   
    
}

}} // namespace mtl::matrix


#endif // MTL_MATRIX_CUPPEN_INCLUDE

