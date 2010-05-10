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
#include <limits>
#include <boost/numeric/mtl/matrix/strict_upper.hpp>
#include <boost/numeric/mtl/operation/diagonal.hpp>
#include <boost/numeric/mtl/operation/one_norm.hpp>
#include <boost/numeric/mtl/operation/sub_matrix.hpp>
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
    size_type        ncols = num_cols(A), nrows = num_rows(A), loops, i(0),row,col;
    value_type       zero= math::zero(A[0][0]), one= math::one(A[0][0]);
    
    double 	     err(numeric_limits<double>::max()), e, f;

    if ( nrows < ncols || nrows > ncols) {
	col= nrows; row= ncols;
    } else {
	col= ncols; row= nrows;
    }
    //init
    Matrix Q(row,row),  R(row,col),  V(A), VT(row,col), E(row,col), 
	   QT(col,col), RT(col,row), S(col,col), D(row,row);

    loops= 100* std::max(nrows,ncols);
    S= one; D= one; E= zero;

    while (err > tol && i < loops ) {
	boost::tie(QT, RT)= qr((V));
 	S*= QT;
	VT= trans(RT);
	boost::tie(Q, R)= qr((VT));
	D*= Q;
	E= triu(R,1);
	V= trans(R);

	//ready for exit when upper(R)=0
	f= two_norm(diagonal(R));
	e= one_norm(E);
	if ( f== zero ) f= 1;
	err= e/f;
	i++;
    } //end while
 
    if ( nrows > ncols ) {
 	col= ncols; row= nrows;
    }

    //fix signs in V
    V= 0;
    for (size_type i= 0; i < col; i++) {
	V[i][i]= std::abs(R[i][i]);
	if (R[i][i] < zero) {
	    for (size_type j= 0; j < row; j++) {
		    S[j][i]= -S[j][i];
	    }
	}
    }
    //std::cout<< "ready signs \n";
    return boost::make_tuple(S,V,D);
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_SVD_INCLUDE
