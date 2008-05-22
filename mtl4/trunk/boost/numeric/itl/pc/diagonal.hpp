// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_PC_DIAGONAL_INCLUDE
#define ITL_PC_DIAGONAL_INCLUDE

#include <boost/numeric/linear_algebra/inverse.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

#include <boost/numeric/itl/utility/solver_proxy.hpp>

namespace itl { namespace pc {

template <typename Matrix>
class diagonal
{
  public:
    typedef typename mtl::Collection<Matrix>::value_type  value_type;
    typedef typename mtl::Collection<Matrix>::size_type   size_type;
    typedef diagonal                                      self;

    diagonal(const Matrix& A) : inv_diag(num_rows(A))
    {
	MTL_THROW_IF(num_rows(A) != num_cols(A), mtl::matrix_not_square());
	using math::reciprocal;

	for (size_type i= 0; i < num_rows(A); ++i)
	    inv_diag[i]= reciprocal(A[i][i]);
    }

    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	MTL_THROW_IF(size(x) != size(inv_diag), mtl::incompatible_size());
	Vector y(size(x));

	for (size_type i= 0; i < size(inv_diag); ++i)
	    y[i]= inv_diag[i] * x[i];
	return y;
    }

    template <typename Vector>
    Vector adjoint_solve(const Vector& x) const
    {
	return solve(x);
    }



#if 0
    // This is more flexible but less generic as the vector type must support the proxy actively
    // Otherwise it only needs move semantics
    template <typename VectorIn>
    solver_proxy<self, VectorIn> solve(const VectorIn& x) const
    {
	return solver_proxy<self, VectorIn>(*this, x);
    }

    template <typename VectorIn>
    solver_proxy<self, VectorIn, false> adjoint_solve(const VectorIn& x) const
    {
	return solver_proxy<self, VectorIn, false>(*this, x);
    }

    template <typename VectorIn, typename VectorOut>
    void solve(const VectorIn& x, VectorOut& y) const
    {
	MTL_THROW_IF(size(x) != size(y), mtl::incompatible_size());	
	MTL_THROW_IF(size(x) != size(inv_diag), mtl::incompatible_size());	

	for (size_type i= 0; i < size(inv_diag); ++i)
	    y[i]= inv_diag[i] * x[i];
    }

    template <typename VectorIn, typename VectorOut>
    void adjoint_solve(const VectorIn& x, VectorOut& y) const
    {
	solve(x, y);
    }
#endif

  protected:
    mtl::dense_vector<value_type>    inv_diag;
}; 


}} // namespace itl::pc

#endif // ITL_PC_DIAGONAL_INCLUDE
