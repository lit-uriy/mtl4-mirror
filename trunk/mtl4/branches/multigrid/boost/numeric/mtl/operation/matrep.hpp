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

#ifndef MTL_MATRIX_MATREP_INCLUDE
#define MTL_MATRIX_MATREP_INCLUDE

#include <algorithm>


namespace mtl { namespace matrix {

///Returns multiplied view of the original matrix. Rows are reproduced by a factor of row. Columns by a factor of col.
template <typename Matrix>
Matrix inline matrep(const Matrix& A, unsigned row, unsigned col)
{
    typedef typename Collection<Matrix>::size_type size_type;
    size_type      rowB, colB, rowA, colA;
    if (row < 1 && col < 1) throw mtl::logic_error("no repetion");

// 	std::cout<< "sizeof(num_rows(A)=)" << typeid(num_rows(A))<< " \n";
// 	std::cout<< 
// "sizeof(row)" << typeid(row) << " \n";
// 	std::cout<< "sizeof(rowB)" << typeid(rowB) << " \n";
	rowA= num_rows(A);
	colA= num_cols(A);
    rowB= rowA*row;
    colB= colA*col;

    Matrix 	B(rowB, colB);
    for (size_type i = 0; i < rowB; i++){
	for (size_type j = 0; j < colB; j++){
		B[i][j]= A[i%num_rows(A)][j%num_cols(A)];
	}
    }
// 	std::cout<< "B=" << B << "\n";
    return B;
}
}} // namespace mtl::matrix


#endif // MTL_MATRIX_MATREP_INCLUDE
