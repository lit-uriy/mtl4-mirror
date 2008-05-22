// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_PC_ILU_0_INCLUDE
#define ITL_PC_ILU_0_INCLUDE

#include <boost/mpl/bool.hpp>

#include <boost/numeric/linear_algebra/identity.hpp>

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>

#include <boost/numeric/itl/utility/solver_proxy.hpp>

namespace itl { namespace pc {

template <typename Matrix>
class ilu_0
{
  public:
    typedef typename mtl::Collection<Matrix>::value_type  value_type;
    typedef typename mtl::Collection<Matrix>::size_type   size_type;
    typedef ilu_0                                         self;

    // Factorization adapted from Saad
    ilu_0(const Matrix& A)
    {
	factorize(A, typename mtl::category<Matrix>::type()); 
    }

    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	return mtl::upper_tri_solve(U, mtl::lower_tri_solve(L, x, false));
    }

    template <typename Vector>
    Vector adjoint_solve(const Vector& x) const
    {
	return mtl::upper_tri_solve(adjoint(L), mtl::lower_tri_solve(adjoint(U), x), false);
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

    // x = LU y --> y= U^{-1} L^{-1} x
    template <typename VectorIn, typename VectorOut>
    void solve(const VectorIn& x, VectorOut& y) const
    {
	y= mtl::upper_tri_solve(U, mtl::lower_tri_solve(L, x, false));

	VectorIn x2(size(x));
	mtl::lower_tri_solve(L, x, x2, false);
	mtl::upper_tri_solve(U, x2, y);
    }

    // x = (LU)^T y --> y= L^{-T} U^{-T} x
    template <typename VectorIn, typename VectorOut>
    void adjoint_solve(const VectorIn& x, VectorOut& y) const
    {
	y= mtl::upper_tri_solve(adjoint(L), mtl::lower_tri_solve(adjoint(U), x), false);
	VectorIn x2(size(x));
	mtl::lower_tri_solve(adjoint(U), x, x2);
	mtl::upper_tri_solve(adjoint(L), x2, y, false);
    }
#endif

  protected:

    void factorize(const Matrix& A, mtl::tag::dense)
    {
	MTL_THROW_IF(true, logic_error("ILU is not intended for dense matrices"));
    }

    void factorize(const Matrix& A, mtl::tag::sparse)
    {
	sparse_factorize(A, typename OrientedCollection<Matrix>::orientation());
    }
    
    void sparse_factorize(const Matrix& A, mtl::tag::col_major)
    {
	MTL_THROW_IF(true, logic_error("ILU for CCS not implemented yet"));
    }

    void sparse_factorize(const Matrix& A, mtl::tag::row_major)
    {
	MTL_THROW_IF(num_rows(A) != num_cols(A), mtl::matrix_not_square());

        using namespace tag;  using traits::range_generator;  
	using math::min; math::identity;

        typedef typename range_generator<row, Matrix>::type       a_cur_type;    
        typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
        typename traits::col<Matrix>::type                        col_a(a); 
        typename traits::const_value<Matrix>::type                value_a(a); 

	Matrix LU= A;
	const size_type empty= identity(min<size_type>(), size_type());

	mtl::dense_vector<size_type> iw(num_rows(A), empty);	

	a_cur_type ac= begin<row>(a), aend= end<row>(a);
	for (unsigned i= 0; ac != aend; ++ac, ++i) {



	    value_type tmp= zero(w[i]);
	    for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
		tmp+= value_a(*aic) * v[col_a(*aic)];	
	    Assign::update(w[i], tmp);
	}
    }


    Matrix   L, U;
}; 






}} // namespace itl::pc

#include <boost/numeric/itl/pc/ilu_0_aux.hpp>

#endif // ITL_PC_ILU_0_INCLUDE
