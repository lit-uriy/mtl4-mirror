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
    Matrix           S(nrows,nrows), D(ncols,ncols), QT(ncols,ncols), RT(ncols,nrows);
    double 	     err(10000), e, f;
 
    std::cout<< "vor start svd\n";
//     std::cout<< "nrows="<< nrows << "\n";
//     std::cout<< "ncols="<< ncols << "\n";
    //if ( nrows > ncols ) throw mtl::logic_error("underdetermined system, use trans(A) instead of A"); //should work for every dimension
    if ( nrows < ncols ) {
	col= nrows; row= ncols;
    } else {
	col= ncols; row= nrows;
    }
    
    Matrix Q(col,col), R(col,row), V(row,col), VT(row,col), E(row,col);
    if ( nrows < ncols ) 
	VT= trans(A);
    else
	V= A;
    std::cout<< "row="<< row << "\n";
    std::cout<< "col="<< col << "\n";

    //init
    loops= 100* std::max(nrows,ncols);
    S= one; D= one; QT= zero; R= zero; E= zero;
    /// loops at the moment ==3
    while (err > tol && i < loops ) {
	std::cout<< "LOOP=" << i << "\n";
	if ( nrows >= ncols ) { ///quadratic part is ok TODO check for row > col
	    std::cout<< "normal\n";
	    std::cout<< "R=\n"<< R << "\n";
 	    std::cout<< "Q=\n"<< Q << "\n";
	    boost::tie(Q, R)= qr((V));
 	    std::cout<< "R=\n"<< R << "\n";
 	    std::cout<< "Q=\n"<< Q << "\n";
	    S*= Q;
	    V= trans(R);
	    boost::tie(Q, R)= qr((V));
	    D*= Q;
// 	    std::cout<< "D=\n"<< D << "\n";
	    E= triu(R,1);
	    V= trans(R);
	} else {
	    std::cout<< "trans\n";
// 	    std::cout<< "QT=\n"<< QT << "\n";
//  	    std::cout<< "RT=\n"<< RT << "\n";
// 	    std::cout<< "VT=\n"<< VT << "\n";
	    boost::tie(QT, RT)= qr((VT));
	    Q= sub_matrix(QT, 0, col, 0, col);
// 	    R= trans(RT);
 	    std::cout<< "S=\n"<< S << "\n";
 	    S*= trans(Q);
 	    std::cout<< "S=\n"<< S << "\n";
	    VT= RT;
	    boost::tie(QT, RT)= qr((VT));
	    D*= QT;
 	    std::cout<< "D=\n" << D << "\n";
 	    std::cout<< "E=\n" << E << "\n";
 	    std::cout<< "RT=\n" << RT << "\n";
	    E= triu(RT,1);
	    VT= RT;
 	    std::cout<< "VT=" << VT<< "\n";
	}
	//ready for exit when upper(R)=0
	e= one_norm(E);
	f= two_norm(diagonal(R));
	if ( f== zero ) f= 1;
	err= e/f;
	i++;
    } //end while
    
    //fix signs in V
    
//     std::cout<< "V=\n" << V << "\n";
//     std::cout<< "R=\n" << R << "\n";
//     std::cout<< "S=\n" << S << "\n";
    if ( nrows >= ncols ) {
	V= 0;
	for (size_type i= 0; i < col; i++) {
	    V[i][i]= std::abs(R[i][i]);
	    if (R[i][i] < zero) {
		for (size_type j= 0; j < col; j++) {
		    S[j][i]= -S[j][i];
		}
	    }
	}
    } else {
	VT= 0;
        for (size_type i= 0; i < col; i++) {
	    VT[i][i]= std::abs(RT[i][i]);
	    if (RT[i][i] < zero) {
		for (size_type j= 0; j < col; j++) {
		    S[j][i]= -S[j][i];
		}
	    }
	}
    }
    std::cout<< "ready inline svd\n";
//     std::cout<< "S=\n" << S << "\n";
//     std::cout<< "V=\n" << V << "\n";
//     std::cout<< "VT=\n" << VT << "\n";
//     std::cout<< "D=\n" << D << "\n";
//     std::cout<< "A=\n" << S*trans(VT)*trans(D) << "\n";
    if ( nrows >= ncols ) {
	std::cout<< "normal\n";
	return boost::make_tuple(S,V,D);
    } else {
	std::cout<< "trans\n";
	return boost::make_tuple(S,trans(VT),D);
    } 
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_SVD_INCLUDE
