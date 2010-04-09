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

#ifndef MTL_MATRIX_SVD_INCLUDE
#define MTL_MATRIX_SVD_INCLUDE

#include <cmath>
#include <boost/numeric/mtl/matrix/strict_upper.hpp>
#include <boost/numeric/mtl/operation/diagonal.hpp>
#include <boost/numeric/mtl/operation/one_norm.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/operation/two_norm.hpp>
#include <boost/tuple/tuple.hpp>

namespace mtl { namespace matrix {

/// QR-Factorization of matrix A
/// Return A=S*V*D' at the moment only for quadratic matrix A
template <typename Matrix>
boost::tuple<typename mtl::dense2D<typename Collection<Matrix>::value_type>,
	     typename mtl::dense2D<typename Collection<Matrix>::value_type>,
	     typename mtl::dense2D<typename Collection<Matrix>::value_type> >
inline svd(const Matrix& A, double tol)
{
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type        ncols = num_cols(A), nrows = num_rows(A), loops, i(0);
    value_type       zero= math::zero(A[0][0]), one= math::one(A[0][0]);
    Matrix           S(nrows,nrows), V(A), E(nrows,ncols), D(ncols,ncols), Q(ncols,ncols), R(ncols,nrows);
    double 	     err(10000), e, f;
 
    if ( nrows > ncols ) throw mtl::logic_error("underdetermined system, use trans(A) instead of A");
    
    //init
    loops= 100* std::max(nrows,ncols);
    S= one; D= one; V= A;
   
    while (err > tol && i < loops ) {
	boost::tie(Q, R)= qr((V));
	S*= Q;
	V= trans(R);
	boost::tie(Q, R)= qr((V));
	D*= Q;

	//ready for exit when upper(R)=0
	E= triu(R,1);
	e= one_norm(E);
	f= two_norm(diagonal(R));
	if ( f== zero ) f= 1;
	err= e/f;
	i++;
	V= trans(R);
    } //end while
    
    //fix signs in V
    V= 0;
    for (size_type i= 0; i < ncols; i++) {
	V[i][i]= std::abs(R[i][i]);
	if (R[i][i] < zero) {
	    for (size_type j= 0; j < nrows; j++) {
		S[j][i]= -S[j][i];
	    }
	}
    }

    return boost::make_tuple(S,V,D);
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_SVD_INCLUDE
