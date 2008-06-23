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
#include <boost/numeric/linear_algebra/inverse.hpp>

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/operation/lower_trisolve.hpp>
#include <boost/numeric/mtl/operation/upper_trisolve.hpp>


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
	factorize(A, typename mtl::traits::category<Matrix>::type()); 
    }

    // solve x = LU y --> y= U^{-1} L^{-1} x
    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	return mtl::upper_trisolve(U, mtl::lower_trisolve(L, x, false));
    }

    // solve x = (LU)^T y --> y= L^{-T} U^{-T} x
    template <typename Vector>
    Vector adjoint_solve(const Vector& x) const
    {
	return mtl::upper_trisolve(adjoint(L), mtl::lower_trisolve(adjoint(U), x), false);
    }


    Matrix get_L() { return L; }
    Matrix get_U() { return U; }

  protected:

    void factorize(const Matrix& A, mtl::tag::dense)
    {
	MTL_THROW_IF(true, mtl::logic_error("ILU is not intended for dense matrices"));
    }

    void factorize(const Matrix& A, mtl::tag::sparse)
    {
	sparse_factorize(A, typename mtl::OrientedCollection<Matrix>::orientation());
    }
    
    void sparse_factorize(const Matrix& A, mtl::tag::col_major)
    {
	MTL_THROW_IF(true, mtl::logic_error("ILU for CCS not implemented yet"));
    }

    // CRS factorization, sorted entries are required
    void sparse_factorize(const Matrix& A, mtl::tag::row_major)
    {
        using namespace mtl; using namespace mtl::tag;  using mtl::traits::range_generator;  
	using math::reciprocal; 

	MTL_THROW_IF(num_rows(A) != num_cols(A), mtl::matrix_not_square());

	Matrix                                                    LU= A;
        typedef typename range_generator<row, Matrix>::type       cur_type;    
        typedef typename range_generator<nz, cur_type>::type      icur_type;            
        typename mtl::traits::col<Matrix>::type                   col(LU);
        typename mtl::traits::value<Matrix>::type                 value(LU); 

	mtl::dense_vector<value_type>                             inv_dia(num_rows(A));
	cur_type ic= begin<row>(LU), iend= end<row>(LU);
	for (size_type i= 0; ic != iend; ++ic, ++i) {

	    for (icur_type kc= begin<nz>(ic), kend= end<nz>(ic); kc != kend; ++kc) {
		size_type k= col(*kc);
		if (k == i) break;

		value_type aik= value(*kc) * inv_dia[k];
		value(*kc, aik);

		for (icur_type jc= kc + 1; jc != kend; ++jc)
		    value(*jc, value(*jc) - aik * LU[k][col(*jc)]);
		// std::cout << "LU after eliminating A[" << i << "][" << k << "] =\n" << LU;			  
	    }
	    inv_dia[i]= reciprocal(LU[i][i]);
	}

	U= upper(LU); 
	L= strict_lower(LU); 
	    
#if 0
	Matrix LD(num_rows(L), num_rows(L)); LD= 1.0; LD+= L;
	std::cout << "ILU factorization of:\n" << A << "\nL = \n" << L << "\nU = \n" << U << "\nLU = \n" << Matrix(LD*U);
#endif
    }

    Matrix   L, U;
}; 


template <typename Matrix, typename Vector>
Vector solve(const ilu_0<Matrix>& P, const Vector& x)
{
    return P.solve(x);
}

template <typename Matrix, typename Vector>
Vector adjoint_solve(const ilu_0<Matrix>& P, const Vector& x)
{
    return P.adjoint_solve(x);
}


}} // namespace itl::pc

#include <boost/numeric/itl/pc/ilu_0_aux.hpp>

#endif // ITL_PC_ILU_0_INCLUDE
