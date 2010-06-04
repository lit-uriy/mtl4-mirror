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

// With contributions from Cornelius Steinhardt

#ifndef MTL_MATRIX_QR_INCLUDE
#define MTL_MATRIX_QR_INCLUDE

#include <cmath>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/operation/householder.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>


namespace mtl { namespace matrix {


// QR-Factorization of matrix A(m x n)
// Return pair R upper triangel matrix and Q= orthogonal matrix. R and Q are always dense2D
template <typename Matrix, typename MatrixQ, typename MatrixR>
void qr(const Matrix& A, MatrixQ& Q, MatrixR& R)
{
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;
    typedef typename Magnitude<value_type>::type      magnitude_type;
    
    size_type        ncols = num_cols(A), nrows = num_rows(A), mini;
    value_type       ref, zero= math::zero(ref);
    magnitude_type   factor= magnitude_type(2);

    Q= 1;
    if (nrows >= ncols)   // row-wise
	mini= (ncols == nrows) ? ncols - 1 : ncols;
    else                  // col-wise
	mini= (ncols == nrows) ? nrows - 1 : nrows;

    for (size_type i = 0; i < mini; i++) {
	irange r(i, imax); // Intervals [i, n-1]
	dense_vector<value_type>     v(nrows-i, zero), w(nrows-i,zero), tmp(ncols-i, zero), qtmp(nrows, zero);

	// dense_vector<value_type>   w(R[r][i])  //not for compressed2D
	for (size_type j = 0; j < nrows-i; j++) 
	    w[j]= R[j+i][i];
        v= householder_s(w);

	//tmp= -v'*R;
	for (size_type a= 0; a < nrows-i; a++)
	    for (size_type b= 0; b < ncols-i; b++)
		tmp[b]-= v[a] * R[a+i][b+i];

	//R+= 2*v*tmp -> R-= 2*v*(v'*R)
	{	
	    inserter<Matrix, update_plus<value_type> > ins_R(R);
	    for (size_type a= 0; a < nrows-i; a++)
		for (size_type b= 0; b < ncols-i; b++)
		    ins_R[a+i][b+i] << factor * v[a] * tmp[b]; // R is same as input type
	} // destroy ins_R

	//update Q: Q-= 2*(v*Q)*v'
	for (size_type a= 0; a < nrows; a++)
	    for (size_type b= i; b < nrows; b++)
		qtmp[a]+= v[b-i]*Q[a][b]; 

	inserter<Matrix, update_minus<value_type> > ins_Q(Q);
	for (size_type a= 0; a < nrows; a++)
	    for (size_type b= i; b < nrows; b++)
		ins_Q[a][b] << factor * qtmp[a] * v[b-i]; //Q is same as input type
	
    } //end for
}

template <typename Matrix>
std::pair<Matrix, Matrix>
inline qr(const Matrix& A)
{
    Matrix           R(A), Q(num_rows(A),num_rows(A));
    qr(A, Q, R);
    return std::make_pair(Q,R);
}



// QR-Factorization of matrix A
// Return Q and R with A = Q*R   R upper triangle and Q othogonal
template <typename Matrix>
std::pair<typename mtl::dense2D<typename Collection<Matrix>::value_type>,
	  typename mtl::dense2D<typename Collection<Matrix>::value_type> >
inline qr_factors(const Matrix& A)
{
    using std::abs;
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Magnitude<value_type>::type      magnitude_type; // to multiply with 2 not 2+0i
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type        ncols = num_cols(A), nrows = num_rows(A);
    value_type       zero= math::zero(A[0][0]), one= math::one(A[0][0]);

    //evaluation of Q
    Matrix  Q(nrows, nrows), Qk(nrows, nrows), HEL(nrows, ncols), R(nrows, ncols), R_tmp(nrows, ncols);
    Q= one; R= zero; HEL= zero;

    boost::tie(Q, R_tmp)= qr(A);
    R= upper(R_tmp);
   
    return std::make_pair(Q,R);
}

}} // namespace mtl::matrix


#endif // MTL_MATRIX_QR_INCLUDE

