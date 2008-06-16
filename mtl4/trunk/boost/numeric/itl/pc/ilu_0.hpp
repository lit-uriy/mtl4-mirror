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

    // solve x = LU y --> y= U^{-1} L^{-1} x
    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	return mtl::upper_tri_solve(U, mtl::lower_tri_solve(L, x, false));
    }

    // solve x = (LU)^T y --> y= L^{-T} U^{-T} x
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
	throw_if(true, logic_error("ILU is not intended for dense matrices"));
    }

    void factorize(const Matrix& A, mtl::tag::sparse)
    {
	sparse_factorize(A, typename OrientedCollection<Matrix>::orientation());
    }
    
    void sparse_factorize(const Matrix& A, mtl::tag::col_major)
    {
	throw_if(true, logic_error("ILU for CCS not implemented yet"));
    }

    // CRS factorization like in Saad, sorted entries are required
    void sparse_factorize(const Matrix& A, mtl::tag::row_major)
    {
        using namespace tag;  using traits::range_generator;  
	using math::min; math::identity; using math::zero; using math::reciprocal; 

	throw_if(num_rows(A) != num_cols(A), mtl::matrix_not_square());
	const size_type       empty= identity(min<size_type>(), size_type());
	Matrix                LU= A;


        typedef typename range_generator<row, Matrix>::type       cur_type;    
        typedef typename range_generator<nz, cur_type>::type      icur_type;            
        typename traits::col<Matrix>::type                        col(A), col_lu(LU); 
        typename traits::offset<Matrix>::type                     offset(A), offset_lu(LU); 
	mtl::dense_vector<size_type>                              iw(num_rows(A), empty);	

	cur_type ac= begin<row>(A), aend= end<row>(A);
	for (unsigned k= 0; ac != aend; ++ac, ++k) {

	    for (icur_type ic= begin<nz>(ac), iend= end<nz>(ac); ic != iend; ++ic) 
		iw[col(*ic)] = offset(*ic);

	    for (icur_type ic= begin<nz>(ac), iend= end<nz>(ac); ic != iend; ++ic) {
		size_type jrow= col(*ic), j= offset(*ic);
		if (jrow < k) {
		    // Multiplier for jrow (immediately set below diagonal)
		    value_type tl= LU.value_from_offset(j)*= LU.value_from_offset(uptr[k]);
		    //value_type tl= LU.value_from_offset(j) * LU.value_from_offset(uptr[k]);
		    //LU.value_from_offset(j)= tl;
		    
		    // Linear combination
#if 0
		    cur_type jrow_c= begin<row>(LU); jrow_c+= jrow;
		    icur_type jjc= begin<nz>(jrow_c), jjend= end<nz>(jrow_c); 
		    while (col_lu(*jjc) <= jrow) ++jjc;  // go behind diagonal
#endif
		    // Linear combination: From behind diagonal (by offset) till end of row
		    for (icur_type jjc(LU, uptr[jrow]+1), jjend(LU, jrow+1, 0); jjc != jjend; ++jjc) {
			size_type jw= iw[col_lu(*jjc)];
			if (jw != empty) 
			    LU.value_from_offset(jw)-= tl * LU.value_from_offset(offset_lu(*jjc));
		    }
		} else {
		    uptr[k]= j;
		    throw_if(jrow != k, mtl::logic_error("No diagonal in ILU_0"));
		    value_type &dia= LU.value_from_offset(j);
		    throw_if(dia == zero(dia), mtl::logic_error("Zero diagonal in ILU_0"));
		    dia= reciprocal(dia); 
		    break;
		}
	    }

	    // Reset iw entries to empty
	    for (icur_type ic= begin<nz>(ac), iend= end<nz>(ac); ic != iend; ++ic) 
		iw[col(*ic)] = empty;
	}
	L= upper(LU); crop(L);
	U= strict_lower(LU); crop(U);
    }


    Matrix   L, U;
}; 






}} // namespace itl::pc

#include <boost/numeric/itl/pc/ilu_0_aux.hpp>

#endif // ITL_PC_ILU_0_INCLUDE
