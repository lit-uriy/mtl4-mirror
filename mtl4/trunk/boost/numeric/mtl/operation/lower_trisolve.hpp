// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_LOWER_TRISOLVE_INCLUDE
#define MTL_LOWER_TRISOLVE_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/adjust_cursor.hpp>

#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl {


namespace detail {

    template <typename Matrix, typename Vector>
    Vector lower_trisolve(const Matrix& A, const Vector& v, bool explicit_diagonal, tag::row_major)
    {
	using namespace tag; using traits::range_generator; using math::one;

	typedef typename Collection<Matrix>::value_type           value_type;
	typedef typename range_generator<row, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::col<Matrix>::type                        col_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	Vector result(v);

	a_cur_type ac= begin<row>(A), aend= end<row>(A); 
	for (int r= 0; ac != aend; ++r, ++ac) {
	    a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac);
	    adjust_cursor(r - num_rows(A) + (explicit_diagonal ? 1 : 0), aiend, typename traits::category<Matrix>::type());
	    MTL_THROW_IF(explicit_diagonal && (aic == aiend || col_a(*--aiend) != r), missing_diagonal());

	    value_type dia= explicit_diagonal ? value_a(*aiend) : one(value_type());
	    typename Collection<Vector>::value_type rr= result[r];

	    for (; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(col_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		rr-= value_a(*aic) * result[col_a(*aic)];
	    }
	    result[r]= rr/= dia;
	}
	return result;
    }	


    template <typename Matrix, typename Vector>
    Vector inline lower_trisolve(const Matrix& A, const Vector& v, bool explicit_diagonal, tag::col_major)
    {
	using namespace tag; using traits::range_generator; 

	typedef typename range_generator<col, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::row<Matrix>::type                        row_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	Vector result(v);

	a_cur_type ac= begin<col>(A), aend= end<col>(A); 
	for (int r= 0; ac != aend; ++r, ++ac) {
	    a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac);
	    adjust_cursor(r + (explicit_diagonal ? 0 : 1), aic, typename traits::category<Matrix>::type());

	    MTL_THROW_IF(explicit_diagonal && (aic == aiend || row_a(*aic) != r), missing_diagonal());
	    typename Collection<Vector>::value_type rr= explicit_diagonal ? (result[r]/= value_a(*aic)) : result[r];

	    if (explicit_diagonal) ++aic;
	    for (; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(row_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		result[row_a(*aic)]-= value_a(*aic) * rr;
	    }
	}
	return result;
    }
}

template <typename Matrix, typename Vector>
Vector inline lower_trisolve(const Matrix& A, const Vector& v, bool explicit_diagonal= true)
{
    // std::cout << "Lower trisolve: A = \n" << A;
    MTL_THROW_IF(num_rows(A) != num_cols(A), matrix_not_square());
    return detail::lower_trisolve(A, v, explicit_diagonal, typename OrientedCollection<Matrix>::orientation());
}


} // namespace mtl

#endif // MTL_LOWER_TRISOLVE_INCLUDE
