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

#ifndef MTL_MATRIX_QUADMATRIX_INCLUDE
#define MTL_MATRIX_QUADMATRIX_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/mtl.hpp>

namespace mtl { namespace matrix {

///Returns matrix with squared entrys
template <typename Matrix>
Matrix inline quadMatrix(const Matrix& A)
{
    typedef typename Collection<Matrix>::size_type size_type;
    Matrix 	S(num_rows(A), num_cols(A));
    S= 0;

    for (size_type i = 0; i < num_rows(A); i++){
	for (size_type j = 0; j < num_cols(A); j++){
		S[i][j]= A[i][j]*A[i][j];	
	}
    }
    return S;
}


///Returns matrix with square roots of entrys
template <typename Matrix>
Matrix inline sqrtMatrix(const Matrix& A)
{
    typedef typename Collection<Matrix>::size_type size_type;
    Matrix 	S(num_rows(A), num_cols(A));
    S= 0;

    for (size_type i = 0; i < num_rows(A); i++){
	for (size_type j = 0; j < num_cols(A); j++){
		S[i][j]= sqrt(A[i][j]);	
	}
    }
    return S;
}

}} // namespace mtl::matrix


#endif // MTL_MATRIX_QUADMATRIX_INCLUDE
