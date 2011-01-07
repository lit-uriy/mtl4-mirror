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
template <typename Matrix, typename Vector>
void inline cuppen(const Matrix& A, Matrix& Q, Matrix& L, Vector& p)
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
std::cout<< "start p=" << p <<"\n";
     dense_vector<value_type>    v(nrows, zero), v1(nrows, zero), diag(nrows, zero), lambda(nrows, zero);
     Vector     perm(nrows, zero), permdiag(nrows, zero);;
    
    if (ncols == 1){
      L[0][0]= A[0][0];
      Q[0][0]= 1;
    } else {
      m= size_type(nrows/2);
      n= nrows - m;
      Vector   perm1(m, zero), perm2(n, zero), perm_intern(nrows, zero);
      Matrix           T1(m, m), T2(n, n), Q1(m,m),Q2(n,n),L1(m,m), L2(n,n);
      //DIVIDE
      value_type b(A[m-1][m]);
      T1= sub_matrix(A, 0, m, 0, m);
      T1[m-1][m-1]-= abs(b);
      std::cout<< "n=" << n <<"\n";
      T2= sub_matrix(A, m, nrows, m, nrows);
      T2[0][0]-= abs(b);

      std::cout << "A is\n" << A;
      std::cout << "T1 is\n" << T1;
      std::cout << "T2 is\n" << T2;



//       std::cout<< "b=" << b <<"\n";
      v[m-1]= b > zero ? one : -one;
      v[m]= one;
       std::cout<< "v_=" << v <<"\n";
      for (size_type i = 0; i < m; i++)
	perm1[i]= i;
      for (size_type i = 0; i < n; i++)
	perm2[i]= i;
      cuppen(T1, Q1, L1, perm1);
      cuppen(T2, Q2, L2, perm2);
       std::cout<< "perm1=" << perm1 <<"\n";
       std::cout<< "perm2=" << perm2 <<"\n";
      mtl::matrix::traits::permutation<>::type P1= mtl::matrix::permutation(perm1);
      mtl::matrix::traits::permutation<>::type P2= mtl::matrix::permutation(perm2);


      for (size_type i = 0; i < n; i++)
	perm2[i]+= m;
     // perm2+= m;
      perm_intern[irange(0,m)]= perm1;
      perm_intern[irange(m, imax)]= perm2;
      std::cout<< "perm_intern=" << perm_intern << "\n";
      perm= perm_intern;
      
//       std::cout<< "v_2=" << v <<"\n";
 
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
       std::cout << "diag=" << diag << "\n";
       std::cout << "perm=" << perm << "\n";
//       std::cout<< "T=" << T <<"\n";
      //permutation on Matrix T
      mtl::matrix::traits::permutation<>::type P= mtl::matrix::permutation(perm);
     mtl::matrix::traits::permutation<>::type PV= mtl::matrix::permutation(perm_intern);
    //       std::cout << "\nP =\n" << P;    
std::cout << "\nP1 =\n" << P1;   
       std::cout << "\nP2 =\n" << P2;   
      std::cout<< "Q1=" << Q1 <<"\n";
      std::cout<< "Q2=" << Q2 <<"\n";
      Matrix TP( P * A*P );// Q1P( P1 * Q1*P1 ), Q2P( P2 * Q2*P2 );
      std::cout << "\nTP =\n" << TP; 
      if (nrows > 2){
      Matrix Q1P( P1*Q1*P1 ), Q2P( P2*Q2*P2 );
      
        std::cout << "\nQ1P =\n" << Q1P;   
        std::cout << "\nQ2P =\n" << Q2P;   
      }
     
       std::cout << "perm1 =" << perm1 << "\n"; 
       std::cout << "perm2 =" << perm2 << "\n";
//       std::cout << "v =" << v;
//       std::cout<< "Q1=" << Q1 <<"\n";
//       std::cout<< "Q2=" << Q2 <<"\n";
//       std::cout<< "m=" << m <<"\n";
       v[irange(0,m)]= Q1[irange(0,m)][m-1];
       v[irange(m,imax)]=Q2[irange(0,n)][0];
       std::cout<< "PV=" << PV << "\n";
 v1=PV*v;
  
       std::cout << "Q1 is\n" << Q1;
       std::cout << "Q2 is\n" << Q2;

#if 0 // transposed yields the same
       v[irange(0,m)]= trans(Q1[m-1][irange(0,m)]);
       v[irange(m,imax)]= trans(Q2[0][irange(0,n)]);
#endif

      std::cout << "Vector perm is " << perm << '\n';
std::cout << "Vector v1 is " << v1 << '\n';
std::cout << "Vector v is " << v << '\n';
std::cout << "Vector diag is " << diag << '\n';
permdiag=perm;
sort(permdiag, v1);
std::cout << "Vector v1 is " << v1 << '\n';
//        sort(perm, v); //permutation on v
//       std::cout << "QQQ   v =" << v;
//       std::cout<< "diag= " << diag << "\n";
//        std::cout<< "abs(b)= " << abs(b) << "\n";
//       std::cout<< "hallo \n";
//       std::cout<<"roots  ="<< secular(lambda, v, diag, abs(b)) <<"\n";
#ifdef PETERS_TEST
      mtl::vector::secular_f<dense_vector<value_type> > secf(lambda, v, diag, abs(b));
      const double eps= 0.01;
      for (unsigned i= 0; i < size(diag); i++)
	  for (double x= diag[i] - 10*eps; x < diag[i] + 10*eps; x+= eps)
	      std::cout << "f(" << x << ") = " << secf.funk(x) << '\n';
#endif
      lambda= secular(lambda, v1, diag, abs(b));

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
	  
          lambda_i/=two_norm(lambda_i);
	  Q[irange(0, imax)][i]= lambda_i;
	 
      }
       L=mtl::vector::diagonal(lambda);
       std::cout<< "L=\n" << L << "\n";
       std::cout<< "Q=\n" << Q << "\n";
  
     
    }
    
    p=perm;
   
    std::cout<< "end p=" << p <<"\n";
}

}} // namespace mtl::matrix


#endif // MTL_MATRIX_CUPPEN_INCLUDE

