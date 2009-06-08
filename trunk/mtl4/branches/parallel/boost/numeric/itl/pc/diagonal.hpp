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

#ifndef ITL_PC_DIAGONAL_INCLUDE
#define ITL_PC_DIAGONAL_INCLUDE

#include <boost/numeric/linear_algebra/inverse.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

#include <boost/numeric/itl/utility/solver_proxy.hpp>
#include <boost/numeric/itl/pc/block.hpp>

namespace itl { namespace pc {

/// Diagonal Preconditioner
template <typename Matrix>
class diagonal
{
  public:
    typedef typename mtl::Collection<Matrix>::value_type  value_type;
    typedef typename mtl::Collection<Matrix>::size_type   size_type;
    typedef diagonal                                      self;

    /// Constructor takes matrix reference
    explicit diagonal(const Matrix& A) : inv_diag(num_rows(A))
    {
	MTL_THROW_IF(num_rows(A) != num_cols(A), mtl::matrix_not_square());
	using math::reciprocal;

	for (size_type i= 0; i < num_rows(A); ++i)
	    inv_diag[i]= reciprocal(A[i][i]);
    }

    /// Member function solve, better use free function solve
    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	MTL_THROW_IF(size(x) != size(inv_diag), mtl::incompatible_size());
	Vector y(size(x));

	for (size_type i= 0; i < size(inv_diag); ++i)
	    y[i]= inv_diag[i] * x[i];
	return y;
    }

    /// Member function for solving adjoint problem, better use free function adjoint_solve
    template <typename Vector>
    Vector adjoint_solve(const Vector& x) const
    {
	return solve(x);
    }

  protected:
    mtl::dense_vector<value_type>    inv_diag;
}; 

#ifdef MTL_HAS_MPI

/// Diagonal Preconditioner for distributed matrices
template <typename Matrix>
class diagonal<mtl::matrix::distributed<Matrix> >
  : public block<mtl::matrix::distributed<Matrix>, diagonal<Matrix> >
{
    typedef mtl::matrix::distributed<Matrix>      matrix_type;
    typedef block<matrix_type, diagonal<Matrix> > base;
  public:
    /// Constructor takes matrix reference
    explicit diagonal(const matrix_type& A) : base(A) {}
};



#if 0 // Just keep it for educational purposes
template <typename Matrix>
class diagonal<mtl::matrix::distributed<Matrix> >
{
  public:
    typedef mtl::matrix::distributed<Matrix>                   matrix_type;
    typedef typename mtl::Collection<matrix_type>::value_type  value_type;
    typedef typename mtl::Collection<matrix_type>::size_type   size_type;
    typedef diagonal                                           self;

    /// Constructor takes matrix reference
    explicit diagonal(const matrix_type& A) 
      : inv_diag(num_rows(local(A))), col_dist(col_distribution(A))
    {
	using math::reciprocal;

	MTL_THROW_IF(row_distribution(A) != col_dist, incompatible_distribution());
	Matrix const& L= local(A);
	MTL_THROW_IF(num_rows(L) != num_cols(L), mtl::matrix_not_square()); // local matrix must be square

	for (size_type i= 0; i < num_rows(L); ++i)
	    inv_diag[i]= reciprocal(L[i][i]);
    }

    /// Member function solve, better use free function solve
    template <typename Vector>
    Vector solve(const Vector& dist_x) const
    {
	MTL_DEBUG_THROW_IF(col_dist != distribution(dist_x), incompatible_distribution());
	MTL_THROW_IF(size(local(dist_x)) != size(inv_diag), mtl::incompatible_size());
	Vector dist_y(dist_x); // copies distribution as well
	typename mtl::DistributedCollection<Vector>::local_type const& x= local(dist_x);
	typename mtl::DistributedCollection<Vector>::local_type&       y= local(dist_y);
	
	for (size_type i= 0; i < size(inv_diag); ++i)
	    y[i]= inv_diag[i] * x[i];
	return dist_y;
    }

    /// Member function for solving adjoint problem, better use free function adjoint_solve
    template <typename Vector>
    Vector adjoint_solve(const Vector& x) const
    {
	return solve(x);
    }
    
  private:
    mtl::dense_vector<value_type>                      inv_diag;
    typename matrix_type::col_distribution_type const& col_dist;
};
#endif // 0

#endif // MTL_HAS_MPI

/// Solve approximately a sparse system in terms of inverse diagonal
template <typename Matrix, typename Vector>
Vector solve(const diagonal<Matrix>& P, const Vector& x)
{
    return P.solve(x);
}

/// Solve approximately the adjoint of a sparse system in terms of inverse diagonal
template <typename Matrix, typename Vector>
Vector adjoint_solve(const diagonal<Matrix>& P, const Vector& x)
{
    return P.adjoint_solve(x);
}


}} // namespace itl::pc

#endif // ITL_PC_DIAGONAL_INCLUDE
