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

#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl {


namespace detail {


    template <typename Matrix, typename DiaTag>
    struct upper_trisolve_t
    {
	typedef typename Collection<Matrix>::value_type           value_type;
	typedef typename OrientedCollection<Matrix>::orientation  my_orientation;
	typedef typename traits::category<Matrix>::type           my_category;

	upper_trisolve_t(const Matrix& A) : A(A), value_a(A), col_a(A), row_a(A)
	{     
	    MTL_THROW_IF(num_rows(A) != num_cols(A), matrix_not_square());
	}

	template <typename Vector>
	Vector operator()(const Vector& v)
	{
	    return apply(v, my_orientation());
	}

    private:
	// Initialization for regular and inverse diagonal is the same
	template <typename Cursor>
	void row_init(int r, Cursor& aic, Cursor& aiend, value_type& dia, tag::universe_diagonal)
	{
	    adjust_cursor(r, aic, my_category());
	    MTL_DEBUG_THROW_IF(aic == aiend || col_a(*aic) != r, missing_diagonal());
	    dia= value_a(*aic); ++aic;
	}

	template <typename Cursor>
	void row_init(int r, Cursor& aic, Cursor&, value_type&, tag::unit_diagonal)
	{
	    adjust_cursor(r + 1, aic, my_category());
	}

	void row_update(value_type& res, value_type& rr, const value_type& dia, tag::regular_diagonal)
	{
	    res= rr / dia;
	}

	void row_update(value_type& res, value_type& rr, const value_type& dia, tag::inverse_diagonal)
	{
	    res= rr * dia;
	}

	void row_update(value_type& res, value_type& rr, const value_type& dia, tag::unit_diagonal)
	{
	    res= rr;
	}


	template <typename Vector>
	Vector apply(const Vector& v, tag::row_major)
	{
	    using namespace tag; using traits::range_generator; using math::one;
	    typedef typename range_generator<row, Matrix>::type       ra_cur_type;    
	    typedef typename range_generator<nz, ra_cur_type>::type   ra_icur_type;            

	    Vector result(v);

	    ra_cur_type ac= begin<row>(A), aend= end<row>(A); 
	    for (int r= num_rows(A) - 1; ac != aend--; --r) {
		ra_icur_type aic= begin<nz>(aend), aiend= end<nz>(aend);
		value_type rr= result[r], dia;
		row_init(r, aic, aiend, dia, DiaTag());
		for (; aic != aiend; ++aic) {
		    MTL_DEBUG_THROW_IF(col_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		    rr-= value_a(*aic) * result[col_a(*aic)];
		}
		row_update(result[r], rr, dia, DiaTag());
	    }
	    return result;
	}


	template <typename Vector>
	Vector apply(const Vector& v, tag::col_major)
	{
	    using namespace tag; using traits::range_generator; using math::one;
	    typedef typename range_generator<col, Matrix>::type       ca_cur_type;    
	    typedef typename range_generator<nz, ca_cur_type>::type   ca_icur_type;            

	    Vector result(v);

	    ca_cur_type ac= begin<col>(A), aend= end<col>(A); 
	    for (int r= num_rows(A) - 1; ac != aend--; --r) {
		ca_icur_type aic= begin<nz>(aend), aiend= end<nz>(aend);
		value_type rr;
		col_init(r, aic, aiend, rr, result[r], DiaTag());

		for (; aic != aiend; ++aic) {
		    MTL_DEBUG_THROW_IF(row_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		    result[row_a(*aic)]-= value_a(*aic) * rr;
		}
	    }
	    return result;
	}

	template <typename Cursor>
	void col_init(int r, Cursor& aic, Cursor& aiend, value_type& rr, value_type& res, tag::regular_diagonal)
	{
	    adjust_cursor(r - num_rows(A) + 1, aiend, my_category());
	    MTL_DEBUG_THROW_IF(aic == aiend, missing_diagonal());
	    --aiend;
	    MTL_DEBUG_THROW_IF(row_a(*aiend) != r, missing_diagonal());
	    rr= res/= value_a(*aiend);
	}
	
	template <typename Cursor>
	void col_init(int r, Cursor& aic, Cursor& aiend, value_type& rr, value_type& res, tag::inverse_diagonal)
	{
	    adjust_cursor(r - num_rows(A) + 1, aiend, my_category());
	    MTL_DEBUG_THROW_IF(aic == aiend, missing_diagonal());
	    --aiend;
	    MTL_DEBUG_THROW_IF(row_a(*aiend) != r, missing_diagonal());
	    rr= res*= value_a(*aiend);
	}

	template <typename Cursor>
	void col_init(int r, Cursor& aic, Cursor& aiend, value_type& rr, value_type& res, tag::unit_diagonal)
	{
	    adjust_cursor(r - num_rows(A), aiend, my_category());
	    rr= res;
	}


	const Matrix& A;
	typename traits::const_value<Matrix>::type  value_a; 
	typename traits::col<Matrix>::type          col_a; 
	typename traits::row<Matrix>::type          row_a;
    };






#if 0
    template <typename Matrix, typename Vector>
    Vector upper_trisolve(const Matrix& A, const Vector& v, bool explicit_diagonal, tag::row_major)
    {
	using namespace tag; using traits::range_generator; using math::one;

	typedef typename Collection<Matrix>::value_type           value_type;
	typedef typename range_generator<row, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::col<Matrix>::type                        col_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	Vector result(v);

	a_cur_type ac= begin<row>(A), aend= end<row>(A); 
	for (int r= num_rows(A) - 1; ac != aend--; --r) {
	    a_icur_type aic= begin<nz>(aend), aiend= end<nz>(aend);



	    adjust_cursor(r + (explicit_diagonal ? 0 : 1), aic, typename traits::category<Matrix>::type());
	    MTL_THROW_IF(explicit_diagonal && (aic == aiend || col_a(*aic) != r), missing_diagonal());

	    typename Collection<Vector>::value_type rr= result[r];
	    value_type dia= explicit_diagonal ? value_a(*aic) : one(value_type());




	    if (explicit_diagonal) ++aic;
	    for (; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(col_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		rr-= value_a(*aic) * result[col_a(*aic)];
	    }
	    result[r]= rr/= dia;
	}
	return result;
    }	


    template <typename Matrix, typename Vector>
    Vector inline upper_trisolve(const Matrix& A, const Vector& v, bool explicit_diagonal, tag::col_major)
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
	    adjust_cursor(r - num_rows(A) + (explicit_diagonal ? 1 : 0), aiend, typename traits::category<Matrix>::type());

	    MTL_THROW_IF(explicit_diagonal && (aic == aiend || row_a(*--aiend) != r), missing_diagonal());
	    typename Collection<Vector>::value_type rr= explicit_diagonal ? (result[r]/= value_a(*aiend)) : result[r];

	    for (; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(row_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		result[row_a(*aic)]-= value_a(*aic) * rr;
	    }
	}
	return result;
    }
#endif 


}

template <typename Matrix, typename Vector>
Vector inline upper_trisolve(const Matrix& A, const Vector& v)
{
    // std::cout << "Upper trisolve: A = \n" << A;
    MTL_THROW_IF(num_rows(A) != num_cols(A), matrix_not_square());

    return detail::upper_trisolve_t<Matrix, tag::regular_diagonal>(A)(v);


    // return detail::upper_trisolve(A, v, tag::regular_diagonal, typename OrientedCollection<Matrix>::orientation());
}

template <typename Matrix, typename Vector, typename DiaTag>
Vector inline upper_trisolve(const Matrix& A, const Vector& v, DiaTag)
{
    // std::cout << "Upper trisolve: A = \n" << A;
    MTL_THROW_IF(num_rows(A) != num_cols(A), matrix_not_square());
    // return detail::upper_trisolve(A, v, DiaTag(), typename OrientedCollection<Matrix>::orientation());

    return detail::upper_trisolve_t<Matrix, DiaTag>(A)(v);
}

} // namespace mtl

#endif // MTL_UPPER_TRISOLVE_INCLUDE
