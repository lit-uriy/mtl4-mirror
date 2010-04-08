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
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/tuple/tuple.hpp>


namespace mtl { namespace matrix {

// QR-Factorization of matrix A
// Return A=S*V*D' 
template <typename Matrix>
boost::tuple<typename mtl::dense2D<typename Collection<Matrix>::value_type>,
      typename mtl::dense2D<typename Collection<Matrix>::value_type>,
      typename mtl::dense2D<typename Collection<Matrix>::value_type> >
inline svd(const Matrix& A, double tol)
{
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type        ncols = num_cols(A), nrows = num_rows(A), loops;
    value_type       zero= math::zero(A[0][0]), one= math::one(A[0][0]);
    Matrix           S(nrows,nrows), V(A), VT(ncols,nrows), D(ncols,ncols), a(A), Q(ncols,ncols), R(ncols,nrows), AT(ncols,nrows);
    double 	     err(10000);

    
    //if ( nrows > ncols ) throw mtl::logic_error("underdetermined system, use trans(A) instead of A");
    
    //init
    loops= 100* std::max(nrows,ncols);
    std::cout<< "loops=" << loops << "\n";
    S= one; D= one; VT= trans(A);
    std::cout<< "S=\n" << S << "\n";
    std::cout<< "V=\n" << V << "\n";
    std::cout<< "D=\n" << D << "\n";
   

    for (size_type i = 0; i < 3; i++) {
	std::cout<< "i=" << i << "\n";
	boost::tie(Q, R)= qr((VT));
        std::cout<< "Q=\n" << Q << "\n";
        std::cout<< "R=\n" << R << "\n";
	AT=Q*R;
        std::cout<< "AT=\n" << trans(R)*trans(Q) << "\n";
	
	
	S= S*trans(Q);
	std::cout<< "S=\n" << S << "\n";
	a= Q*R;
	std::cout<< "a=\n" << a << "\n";
	V= V*Q;
	std::cout<< "V=\n" << V << "\n";
	
	//ready for exit???
	

    }
    std::cout<< "Fertig inline\n";
    return boost::make_tuple(S,V,D);
}





}} // namespace mtl::matrix


#endif // MTL_MATRIX_SVD_INCLUDE

