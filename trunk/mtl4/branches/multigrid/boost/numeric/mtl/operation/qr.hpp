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

//TODO docu need to be changed
// QR-Factorization of matrix A
// Return A  with R=triu(A) and L=tril(A,-1) L in form of Householder-vectors
template <typename Matrix>
std::pair<typename mtl::dense2D<typename Collection<Matrix>::value_type>,
	  typename mtl::dense2D<typename Collection<Matrix>::value_type> >
	inline qr(const Matrix& A)
{
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type        ncols = num_cols(A), nrows = num_rows(A), mini;
    value_type       zero= math::zero(A[0][0]);
    Matrix           R(A), Q(nrows,nrows);

    Q= 1;
   if ( nrows < ncols ) throw mtl::logic_error("underdetermined system, use trans(A) instead of A");
    if(ncols == nrows)
	mini= ncols - 1;
    else
	mini= ncols;

    for (size_type i = 0; i < mini; i++) {
	irange r(i, imax); // Intervals [i, n-1]
	dense_vector<value_type>     v(nrows-i, zero), tmp(ncols-i, zero), qtmp(nrows, zero), w(nrows-i, zero);
	
	for (size_type j = 0; j < size(w); j++)
	    w[j]= R[j+i][i];

        v= householder_s(w);

	for (size_type a= 0; a < nrows-i; a++){
		for (size_type b= 0; b < ncols-i; b++){
			tmp[b]-= v[a]*R[a+i][b+i];
		}	
	}

	for (size_type a= 0; a < nrows-i; a++){
		for (size_type b= 0; b < ncols-i; b++){
			R[a+i][b+i]= R[a+i][b+i] + 2 * v[a]*tmp[b]; 
		}
	}

	//update Q
	for (size_type a= 0; a < nrows; a++){
		for (size_type b= i; b < nrows; b++){
			qtmp[a]+= v[b-i]*Q[a][b]; 
		}	
	}

	for (size_type a= 0; a < nrows; a++){
		for (size_type b= i; b < nrows; b++){
			Q[a][b]=Q[a][b]- 2* qtmp[a] * v[b-i]; 
		}	
	}
     
    }
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
   

//     for(size_type i = 0; i < ncols; i++)
//         Q[i][i] = one;
 #if 0   
    for(size_type i = 0; i < nrows-1; i++){
        dense_vector<value_type>     z(nrows-i);
	// z[irange(1, nrows-i-1)]= B[irange(i+1, nrows-1)][i];
        for(size_type k = i+1; k < nrows-1; k++){
            z[k-i]= B[k][i];
        }
        z[0]= one;
	Qk= one;

        Qk= zero;
        for(size_type k = 0; k < ncols; k++){
            Qk[k][k]= one;
        }

	magnitude_type factor= magnitude_type(2) / abs(dot(z, z)); // abs: x+0i -> x
	// Qk[irange(i, nrows)][irange(i, ncols)]-= factor * z[irange(0, nrows-i)] * trans(z[irange(0, ncols-i)]);
        for(size_type row = i; row < nrows; row++){
            for(size_type col = i; col < ncols; col++){
                Qk[row][col]-= factor * z[row-i] * z[col-i]; 
            }
        }
	// Q*= Qk;
        HEL = Q * Qk;
	Q = HEL;
    }
    // R= upper(B);
    //evaluation of R
    for(size_type row = 0; row < ncols; row++){
        for(size_type col = row; col < ncols; col++){
            R[row][col]= B[row][col];
        }
    }
#endif
    return std::make_pair(Q,R);
}

}} // namespace mtl::matrix


#endif // MTL_MATRIX_QR_INCLUDE

