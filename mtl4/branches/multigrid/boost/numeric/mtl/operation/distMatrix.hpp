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

#ifndef MTL_MATRIX_DISTMATRIX_INCLUDE
#define MTL_MATRIX_DISTMATRIX_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/operation/matrep.hpp>
#include <boost/numeric/mtl/operation/quadMatrix.hpp>


namespace mtl { namespace matrix {

///The resulting matrix contains the distances of points of matrix A and the associated point in matrix B
template <typename Matrix>
Matrix inline distMatrix(const Matrix& A, const Matrix& B)
{
    typedef typename Collection<Matrix>::size_type size_type;
    size_type      rowB, colB, rowA, colA;
    
    rowA= num_rows(A);    colA= num_cols(A);
    rowB= num_rows(B);    colB= num_cols(B);

    irange r(0, imax);

    if (colA != colB) throw mtl::logic_error("second dimension of Input is wrong");

    Matrix 	S(rowA, rowB);
    S= 0;

    for (size_type i = 0; i < colA; i++){
	//C(k), D(k)
	Matrix	C(rowA,rowB), D(rowB,rowA), inputA(rowA,1), inputB(rowB,1), T(rowA,rowB);
	for (size_type k = 0; k < rowA; k++){
		inputA[k][0]= A[k][i];
	}
	for (size_type k = 0; k < rowB; k++){
		inputB[k][0]= A[k][i];
	}
	C= 0; D= 0;
	C= matrep(inputA, 1, rowB);
	D= matrep(inputB, 1, rowA);
	T= C - trans(D);
  	S= S+quadMatrix(T);

    }
    S= sqrtMatrix(S);
    return S;
}
}} // namespace mtl::matrix


#endif // MTL_MATRIX_DISTMATRIX_INCLUDE
