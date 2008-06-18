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

namespace mtl {



namespace detail {

    template <typename Matrix, typename Vector>
    Vector lower_trisolve(const Matrix& A, const Vector& v, tag::row_major)
    {
	using namespace tag; using traits::range_generator; 

	typedef typename range_generator<row, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::col<Matrix>::type                        col_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	Vector result(v);

	a_cur_type ac= begin<row>(A), aend= end<row>(A); 
	for (int r= 0; ac != aend; ++r, ++ac) {
	    a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac);
	    adjust_cursor(r - num_rows(A) + 1, aiend, typename traits::category<Matrix>::type());
	    throw_if(aic == aiend || col_a(*--aiend) != r, missing_diagonal());

	    typename Collection<Matrix>::value_type dia= value_a(*aiend);
	    typename Collection<Vector>::value_type rr= result[r];

	    for (; aic != aiend; ++aic) {
		debug_throw_if(col_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		rr-= value_a(*aic) * result[col_a(*aic)];
	    }
	    result[r]= rr/= dia;
	}
	return result;
    }	


    template <typename Matrix, typename Vector>
    Vector inline lower_trisolve(const Matrix& A, const Vector& v, tag::col_major)
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
	    adjust_cursor(r, aic, typename traits::category<Matrix>::type());

	    throw_if(aic == aiend || row_a(*aic) != r, missing_diagonal());
	    typename Collection<Vector>::value_type rr= (result[r]/= value_a(*aic));

	    for (++aic; aic != aiend; ++aic) {
		debug_throw_if(row_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		result[row_a(*aic)]-= value_a(*aic) * rr;
	    }
	}
	return result;
    }
}

template <typename Matrix, typename Vector>
Vector inline lower_trisolve(const Matrix& A, const Vector& v)
{
    throw_if(num_rows(A) != num_cols(A), matrix_not_square());
    return detail::lower_trisolve(A, v, typename OrientedCollection<Matrix>::orientation());
}


} // namespace mtl

#endif // MTL_LOWER_TRISOLVE_INCLUDE
