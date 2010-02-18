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

#ifndef MTL_MATRIX_GIVENS_INCLUDE
#define MTL_MATRIX_GIVENS_INCLUDE

#include <cmath>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/householder.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>


namespace mtl { namespace matrix {

/// one MG_Level
template <typename Matrix, typename Vector>
class mg_level
{
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;

  private:
    Matrix&    A, P, R;
    Vector&    x, b, c;
    int        level;


  public:
    /// Constructor takes %matrix \p A 
    mg_level(Matrix& A, Vector& x, Vector& b, int level) : A(A), x(x), b(b), level(level)
    {
	std::cout<< "mg_level konstruktor" << std::endl;
    }

    /// coarse %matrix \p A'=R*A*P  
    Matrix& coarse_matrix(const Matrix& A, const Matrix& P, const Matrix& R)
    {
	return R*A*P;
    }

    /// coarse %matrix \p A with simple strategy
    Vector& coarse_simple(const Matrix& A, const Vector& b)
    {
	Vector c(num_cols(A));
	c= amg_coarsepoints_simple(A, b);
	return c;
    }

    /// coarse %matrix \p A with default strategy
    Vector& coarse_default(const Matrix& A, const Vector& b)
    {
	Vector c(num_cols(A));
	c= amg_coarsepoints_default(A, b);
	return c;
    }

    /// coarse %matrix \p A with runge-stueben strategy
    Vector& coarse_rs(const Matrix& A, const Vector& b, double beta)
    {
	Vector c(num_cols(A));
	c= amg_coarsepoints_rs(A, b, beta);
	return c;
    }

    /// coarse %matrix \p A with notay strategy
    Vector& coarse_notay(const Matrix& A, const Vector& b, double beta)
    {
	Vector c(num_cols(A));
	c= amg_coarsepoints_notay(A, b, beta);
	return c;
    }

    /// prolonagtion %matrix \p P
    Matrix& prolongate(const Matrix& A, const Vector& c, double beta)
    {
	return amg_prolongate(A, c, beta);
    }

    /// prolonagtion %matrix \p P with F-smoothing
    Matrix& prolongate_f(const Matrix& A, const Vector& b,  const Vector& c)
    {
	return amg_prolongate_f(A, b, c);
    }

    /// restriktion %matrix \p R with average interpolation
    Matrix& restrict_average(const Matrix& A, const Vector& c, double beta)
    {
	return amg_restict_average(A, c, beta);
    }	
	
    /// restriktion %matrix \p R with omitting the fine grid points 
    Matrix& restrict_simple(const Matrix& A, const Vector& c, double beta)
    {
	return amg_restict_simple(A, c, beta);
    }

  
};

}} // namespace mtl::matrix

#endif // MTL_MATRIX_GIVENS_INCLUDE
