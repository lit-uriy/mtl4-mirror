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

#ifndef MTL_MATRIX_LU_INCLUDE
#define MTL_MATRIX_LU_INCLUDE

#include <cmath>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/upper.hpp>
#include <boost/numeric/mtl/matrix/lower.hpp>
#include <boost/numeric/mtl/matrix/permutation.hpp>
#include <boost/numeric/mtl/operation/lower_trisolve.hpp>
#include <boost/numeric/mtl/operation/upper_trisolve.hpp>
#include <boost/numeric/mtl/operation/max_pos.hpp>
#include <boost/numeric/mtl/operation/swap_row.hpp>


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

/// LU factorization in place (with pivoting and without optimization so far)
template <typename Matrix, typename PermuationVector>
void inline lu(Matrix& A, PermuationVector& P)
{
    using math::zero;
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type ncols = num_cols(A), nrows = num_rows(A);

    MTL_THROW_IF(ncols != nrows , matrix_not_square());
    P.change_dim(nrows);

    for (size_type i= 0; i < nrows; i++)
        P[i]= i;

    for (size_type i= 0; i < nrows; i++){

	irange r(i+1, imax), ir(i, i+1); // Intervals [i+1, n-1], [i, i]
	size_type rmax= max_abs_pos(A[irange(i, imax)][ir]).first + i;
	swap_row(A, i, rmax); 
	swap_row(P, i, rmax);
	

	MTL_THROW_IF(A[i][i] == zero(A[i][i]), runtime_error("Singular matrix (0 in pivot column)"));
        // Scale column i
	A[r][ir]/= A[i][i];
	// Decrease bottom right block of matrix
	A[r][r]-= A[r][ir] * A[ir][r];
	// std::cout << "After exchanging " << i << " and " << rmax << ", A is \n" << A;
    }
    // std::cout << "permuation \n" << P << "\n";
}


#if 0
// For illustration purposes
template <typename Matrix>
void inline lu(Matrix& LU)
{
    MTL_THROW_IF(num_rows(LU) != num_cols(LU), matrix_not_square());

    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;

    size_type n= num_rows(LU);
    for (size_type k= 0; k < num_rows(LU); k++) {
    value_type pivot= LU[k][k];
    for (size_type j= k+1; j < n; j++) {
        value_type alpha= LU[j][k]/= pivot;
        for (size_type i= k+1; i < n; i++)
        LU[j][i]-= alpha * LU[k][i];
    }
    }
}
#endif


/// LU factorization without factorization that returns the matrix
template <typename Matrix>
Matrix inline lu_f(const Matrix& A)
{
    Matrix LU(A);
    lu(LU);
    return LU;
}

/// Solve Ax = b by LU factorization without pivoting; vector x is returned
template <typename Matrix, typename Vector>
Vector inline lu_solve_straight(const Matrix& A, const Vector& b)
{
    Matrix LU(A);
    lu(LU);
    return upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), b));
}

/// Apply the factorization L*U with permutation P on vector b
template <typename Matrix, typename PermVector, typename Vector>
Vector inline lu_apply(const Matrix& LU, const PermVector& P, const Vector& b)
{
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type ncols = num_cols(LU), nrows = num_rows(LU);
    MTL_THROW_IF(nrows != ncols , matrix_not_square());

    Vector                    bp(nrows);
    for (size_type i= 0; i < nrows; i++)
        bp[i] = b[P[i]];

    return upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), bp));
}


/// Solve Ax = b by LU factorization with column pivoting; vector x is returned
template <typename Matrix, typename Vector>
Vector inline lu_solve(const Matrix& A, const Vector& b)
{
    typedef typename Collection<Matrix>::size_type    size_type;
    dense_vector<std::size_t> P(num_rows(A));
    Matrix                    LU(A);

    lu(LU, P);
    return lu_apply(LU, P, b);
}




}} // namespace mtl::matrix 

















// ###
#if 0

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

#if 0
// For illustration purposes
template <typename Matrix>
void inline lu(Matrix& LU)
{
    MTL_THROW_IF(num_rows(LU) != num_cols(LU), matrix_not_square());

    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;

    size_type n= num_rows(LU);
    for (size_type k= 0; k < num_rows(LU); k++) {
	value_type pivot= LU[k][k];
	for (size_type j= k+1; j < n; j++) {
	    value_type alpha= LU[j][k]/= pivot;
	    for (size_type i= k+1; i < n; i++)
		LU[j][i]-= alpha * LU[k][i];
	}
    }
}
#endif


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

#endif

#endif // MTL_MATRIX_LU_INCLUDE
