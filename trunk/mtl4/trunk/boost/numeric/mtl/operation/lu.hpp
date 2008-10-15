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

#ifndef MTL_MATRIX_LU_INCLUDE
#define MTL_MATRIX_LU_INCLUDE

#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/upper.hpp>
#include <boost/numeric/mtl/matrix/lower.hpp>
#include <boost/numeric/mtl/operation/lower_trisolve.hpp>
#include <boost/numeric/mtl/operation/upper_trisolve.hpp>


namespace mtl { namespace matrix {


/// LU factorization in place (without pivoting and optimization so far)
template <typename Matrix>
void inline lu(Matrix& LU)
{
    MTL_THROW_IF(num_rows(LU) != num_cols(LU), matrix_not_square());

    for (std::size_t k= 0; k < num_rows(LU); k++) {	
	irange r(k+1, imax), kr(k, k+1); // Intervals [k+1, n-1], [k, k]
	LU[r][kr]/= LU[k][k];
	LU[r][r]-= LU[r][kr] * LU[kr][r];
    }
}

/// LU factorization that returns the matrix
template <typename Matrix>
Matrix inline lu_f(const Matrix& A)
{
    Matrix LU(A);
    lu(LU);
    return LU;
}

template <typename Matrix, typename Vector>
Vector inline lu_solve(const Matrix& A, const Vector& v)
{
    Matrix LU(A);
    lu(LU);
    return upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), v));
}


}} // namespace mtl::matrix

#endif // MTL_MATRIX_LU_INCLUDE
