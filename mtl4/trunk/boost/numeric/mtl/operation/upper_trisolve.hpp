// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_UPPER_TRISOLVE_INCLUDE
#define MTL_UPPER_TRISOLVE_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/adjust_cursor.hpp>

namespace mtl {


namespace detail {

    template <typename Matrix, typename Vector>
    Vector upper_trisolve(const Matrix& A, const Vector& v, tag::row_major)
    {
	using namespace tag; using traits::range_generator; 

	typedef typename range_generator<row, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::col<Matrix>::type                        col_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	Vector result(v);

	a_cur_type ac= begin<row>(A), aend= end<row>(A); 
	for (int r= num_rows(A) - 1; ac != aend--; --r) {
	    a_icur_type aic= begin<nz>(aend), aiend= end<nz>(aend);
	    adjust_cursor(r, aic, typename traits::category<Matrix>::type());
	    throw_if(aic == aiend || col_a(*aic) != r, missing_diagonal());

	    typename Collection<Matrix>::value_type dia= value_a(*aic);
	    typename Collection<Vector>::value_type rr= result[r];

	    for (++aic; aic != aiend; ++aic) {
		debug_throw_if(col_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		rr-= value_a(*aic) * result[col_a(*aic)];
	    }
	    result[r]= rr/= dia;
	}
	return result;
    }	


    template <typename Matrix, typename Vector>
    Vector inline upper_trisolve(const Matrix& A, const Vector& v, tag::col_major)
    {
	using namespace tag; using traits::range_generator; 

	typedef typename range_generator<col, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::row<Matrix>::type                        row_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	Vector result(v);

	a_cur_type ac= begin<col>(A), aend= end<col>(A); 
	for (int r= num_rows(A) - 1; ac != aend--; --r) {
	    a_icur_type aic= begin<nz>(aend), aiend= end<nz>(aend);
	    adjust_cursor(r - num_rows(A) + 1, aiend, typename traits::category<Matrix>::type());

	    throw_if(aic == aiend || row_a(*--aiend) != r, missing_diagonal());
	    typename Collection<Vector>::value_type rr= (result[r]/= value_a(*aiend));

	    for (; aic != aiend; ++aic) {
		debug_throw_if(row_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		result[row_a(*aic)]-= value_a(*aic) * rr;
	    }
	}
	return result;
    }
}

template <typename Matrix, typename Vector>
Vector inline upper_trisolve(const Matrix& A, const Vector& v)
{
    throw_if(num_rows(A) != num_cols(A), matrix_not_square());
    return detail::upper_trisolve(A, v, typename OrientedCollection<Matrix>::orientation());
}

} // namespace mtl

#endif // MTL_UPPER_TRISOLVE_INCLUDE
