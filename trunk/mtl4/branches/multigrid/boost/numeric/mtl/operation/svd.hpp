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
#include <boost/numeric/mtl/operation/sub_matrix.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/operation/two_norm.hpp>
#include <boost/tuple/tuple.hpp>

namespace mtl { namespace matrix {

/// QR-Factorization of matrix A
/// Return A=S*V*D' at the moment only for quadratic matrix A/// working on rectangular 
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
    
    double 	     err(10000), e, f;
 
//     if ( nrows > ncols ) throw mtl::logic_error("underdetermined system, use trans(A) instead of A"); //should work for every dimension
    if ( nrows < ncols ) {
	col= nrows; row= ncols;
    } else {
	col= ncols; row= nrows;
    }
     Matrix Q(row,row), R(row,col), V(col,row), VT(row,col), 
	    E(row,col), QT(col,col), RT(col,row), S(row,row),ST(col,col),D(row,row);
    std::cout<< "A=\n" << A << "\n"; std::cout<< "V=\n" << V << "\n";
    V= A;
    //init
    loops= 100* std::max(nrows,ncols);
    ST= one; S= one; D= one; E= zero;
    while (err > tol && i < loops ) {
	if ( nrows >= ncols ) { 
	    boost::tie(Q, R)= qr((V));
 	    S*= Q;
	    V= trans(R);
	    boost::tie(Q, R)= qr((V));
	} else {
	    boost::tie(QT, RT)= qr((V));
	    ST*= QT;
	    VT= trans(RT);
            boost::tie(Q, R)= qr((VT));
 	}
        D*= Q;
	E= triu(R,1);
	V= trans(R);

	//ready for exit when upper(R)=0
	f= two_norm(diagonal(R));
	e= one_norm(E);
	
	if ( f== zero ) f= 1;
	err= e/f;
	i++;
// 	std::cout<< "e=" << e << "  und f=" << f << "   ERROR= " << err << "\n";
    } //end while
 
    //fix signs in V
    
    V= 0;
    for (size_type i= 0; i < col; i++) {
	V[i][i]= std::abs(R[i][i]);
	if (R[i][i] < zero) {
	    for (size_type j= 0; j < col; j++) {
		if ( nrows >= ncols ) {
		    S[j][i]= -S[j][i];
		} else {
		    ST[j][i]= -ST[j][i];
		}
	    }
	}
    }
    if ( nrows >= ncols )
	return boost::make_tuple(S,V,D);
    else 
	return boost::make_tuple(ST,V,D);
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_SVD_INCLUDE
