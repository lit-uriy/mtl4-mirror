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
    Matrix           R(A), Q(ncols,ncols);

    Q= 1;
    std::cout<< "Q=\n" << Q << "\n";
    if ( nrows < ncols ) throw mtl::logic_error("underdetermined system, use trans(A) instead of A");
    if(ncols == nrows)
	mini= nrows - 1;
    else
	mini= nrows;

    for (size_type i = 0; i < mini; i++) {
	irange r(i, imax); // Intervals [i, n-1]
	dense_vector<value_type>     v(nrows-i, zero), tmp(nrows-i, zero), qtmp(nrows, zero), w(nrows-i, zero);

	for (size_type j = 0; j < size(w); j++)
	    w[j]= R[j+i][i];
        v= householder_s(w);
	std::cout<< "v=" << v << "\n";


	//work for monday  do rectangel matrix
	for (size_type a= 0; a < nrows-i; a++){
		for (size_type b= 0; b < ncols-i; b++){
			tmp[a]-= v[b]*R[b+i][a+i];   //chaNGE 	a and b
		}	
	}
	std::cout<< "tmp=" << tmp << "\n";
	for (size_type a= 0; a < ncols-i; a++){
		for (size_type b= 0; b < nrows-i; b++){
			R[a+i][b+i]= R[a+i][b+i] + 2 * v[a]*tmp[b]; 
		}
	}
	
	std::cout<< "vor update Q\n";
	//update Q
	for (size_type a= 0; a < nrows; a++){
		for (size_type b= i; b < ncols; b++){
// 			std::cout<< "Q["<< a << "][" << b << "]=" << Q[a][b] << "\n";
			qtmp[a]+= v[b-i]*Q[a][b]; 
		}	
	}
	std::cout<< "qtemp= " << qtmp << "\n";
	
	for (size_type a= 0; a < nrows; a++){
		for (size_type b= i; b < ncols; b++){
//   			std::cout<< "Q["<< a << "][" << b << "]=" << Q[a][b] << "\n";
			Q[a][b]=Q[a][b]- 2* qtmp[a] * v[b-i]; 
		}	
	}
	std::cout << "Q=\n" << Q << "\n";
	std::cout << "R=\n" << R << "\n";

      
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
    Matrix  Q(nrows, ncols), Qk(ncols, ncols), HEL(nrows, ncols), R(ncols, ncols), B(nrows, ncols);
    Q= one; R= zero; HEL= zero;

    B= qr(A);

//     for(size_type i = 0; i < ncols; i++)
//         Q[i][i] = one;
    
    for(size_type i = 0; i < nrows-1; i++){
        dense_vector<value_type>     z(nrows-i);
	// z[irange(1, nrows-i-1)]= B[irange(i+1, nrows-1)][i];
        for(size_type k = i+1; k < nrows-1; k++){
            z[k-i]= B[k][i];
        }
        z[0]= one;
	Qk= one;
#if 0
        Qk= zero;
        for(size_type k = 0; k < ncols; k++){
            Qk[k][k]= one;
        }
#endif
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
    return std::make_pair(Q,R);
}

}} // namespace mtl::matrix


#endif // MTL_MATRIX_QR_INCLUDE

