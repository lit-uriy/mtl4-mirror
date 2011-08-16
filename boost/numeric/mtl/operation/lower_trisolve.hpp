// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
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
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>

namespace mtl { namespace matrix {


namespace detail {

    template <typename Value>
    Value inline lower_trisolve_diavalue(const Value& v, tag::regular_diagonal)
    {
	using math::reciprocal;
	return reciprocal(v);
    }

    template <typename Value>
    Value inline lower_trisolve_diavalue(const Value& v, tag::inverse_diagonal)
    {
	return v;
    }    

    template <typename Matrix, typename VectorIn, typename VectorOut>
    inline void lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w, tag::row_major, tag::unit_diagonal)
    {
	vampir_trace<5022> tracer;
	namespace traits = mtl::traits;
	using namespace tag; using traits::range_generator; using math::one; using mtl::detail::adjust_cursor;

	typedef typename Collection<Matrix>::value_type           value_type;
	typedef typename Collection<Matrix>::size_type            size_type;
	typedef typename range_generator<row, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::col<Matrix>::type                        col_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	w= v;
	a_cur_type ac= begin<row>(A), aend= end<row>(A);
	++ac;
	for (size_type r= 1; ac != aend; ++r, ++ac) {
	    a_icur_type aic= begin<nz>(ac), aiend= lower_bound<nz>(ac, r);
	    typename Collection<VectorOut>::value_type rr= w[r];

	    for (; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(col_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		rr-= value_a(*aic) * w[col_a(*aic)];
	    }
	    w[r]= rr;
	}
    }	


    template <typename Matrix, typename VectorIn, typename VectorOut, typename DiaTag>
    inline void lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w, tag::row_major,
				 DiaTag)
    {
	vampir_trace<5022> tracer;
	namespace traits = mtl::traits;
	using namespace tag; using traits::range_generator; using math::one; using mtl::detail::adjust_cursor;

	typedef typename Collection<Matrix>::value_type           value_type;
	typedef typename range_generator<row, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename traits::col<Matrix>::type                        col_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	w= v;
	a_cur_type ac= begin<row>(A), aend= end<row>(A); 
	for (typename Collection<Matrix>::size_type r= 0; ac != aend; ++r, ++ac) {
	    a_icur_type aic= begin<nz>(ac), aiend= lower_bound<nz>(ac, r+1);
	    MTL_THROW_IF(aic == aiend, missing_diagonal());
	    --aiend;
	    MTL_THROW_IF(col_a(*aiend) != r, missing_diagonal());

	    value_type dia= value_a(*aiend);
	    typename Collection<VectorOut>::value_type rr= w[r];

	    for (; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(col_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		rr-= value_a(*aic) * w[col_a(*aic)];
	    }
	    w[r]= rr * lower_trisolve_diavalue(dia, DiaTag());
	}
    }	


    template <typename Matrix, typename VectorIn, typename VectorOut>
    inline void lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w, tag::col_major, tag::unit_diagonal)
    {
	vampir_trace<5022> tracer;
	using namespace tag; using mtl::traits::range_generator; using mtl::detail::adjust_cursor;

	typedef typename range_generator<col, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename mtl::traits::row<Matrix>::type                        row_a(A); 
	typename mtl::traits::const_value<Matrix>::type                value_a(A); 

	w= v;
	a_cur_type ac= begin<col>(A), aend= end<col>(A); 
	for (typename Collection<Matrix>::size_type r= 0; ac != aend; ++r, ++ac) {
	    a_icur_type aic= lower_bound<nz>(ac, r+1), aiend= end<nz>(ac);
	    typename Collection<VectorOut>::value_type rr= w[r];

	    for (; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(row_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		w[row_a(*aic)]-= value_a(*aic) * rr;
	    }
	}
    }

    template <typename Matrix, typename VectorIn, typename VectorOut, typename DiaTag>
    inline void lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w, tag::col_major, DiaTag)
    {
	vampir_trace<5022> tracer;
	using namespace tag; using mtl::traits::range_generator; using mtl::detail::adjust_cursor;

	typedef typename range_generator<col, Matrix>::type       a_cur_type;    
	typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename mtl::traits::row<Matrix>::type                        row_a(A); 
	typename mtl::traits::const_value<Matrix>::type                value_a(A); 

	w= v;
	a_cur_type ac= begin<col>(A), aend= end<col>(A); 
	for (typename Collection<Matrix>::size_type r= 0; ac != aend; ++r, ++ac) {
	    a_icur_type aic= lower_bound<nz>(ac, r), aiend= end<nz>(ac);
	    MTL_DEBUG_THROW_IF(aic == aiend || row_a(*aic) != r, missing_diagonal());
	    typename Collection<VectorOut>::value_type rr= w[r]*= lower_trisolve_diavalue(value_a(*aic), DiaTag());

	    for (++aic; aic != aiend; ++aic) {
		MTL_DEBUG_THROW_IF(row_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		w[row_a(*aic)]-= value_a(*aic) * rr;
	    }
	}
    }


}  // detail


/// Solves the lower triangular matrix A  with the rhs v and returns the solution vector
template <typename Matrix, typename Vector>
Vector inline lower_trisolve(const Matrix& A, const Vector& v)
{
    Vector w(resource(v));
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), tag::regular_diagonal());
    return w;
}

/// Solves the lower triangular matrix A  with the rhs v 
template <typename Matrix, typename VectorIn, typename VectorOut>
inline void lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w)
{
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), tag::regular_diagonal());
}

/// Solves the lower triangular matrix A (only one's in the diagonal) with the rhs v and returns the solution vector
template <typename Matrix, typename Vector>
Vector inline unit_lower_trisolve(const Matrix& A, const Vector& v)
{
    Vector w(resource(v));
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), tag::unit_diagonal());
    return w;
}

/// Solves the lower triangular matrix A (only one's in the diagonal) with the rhs v and returns the solution vector
template <typename Matrix, typename VectorIn, typename VectorOut>
inline void unit_lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w)
{
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), tag::unit_diagonal());
}

/// Solves the lower triangular matrix A (inverse the diagonal) with the rhs v and returns the solution vector
template <typename Matrix, typename Vector>
Vector inline inverse_lower_trisolve(const Matrix& A, const Vector& v)
{
    Vector w(resource(v));
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), tag::inverse_diagonal());
    return w;
}

/// Solves the lower triangular matrix A (inverse the diagonal) with the rhs v and returns the solution vector
template <typename Matrix, typename VectorIn, typename VectorOut>
inline void inverse_lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w)
{
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), tag::inverse_diagonal());
}

template <typename Matrix, typename Vector, typename DiaTag>
Vector inline lower_trisolve(const Matrix& A, const Vector& v, DiaTag)
{
    Vector w(resource(v));
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), DiaTag());
    return w;
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename DiaTag>
inline void lower_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w, DiaTag)
{
    detail::lower_trisolve(A, v, w, typename OrientedCollection<Matrix>::orientation(), DiaTag());
}



}} // namespace mtl::matrix

#endif // MTL_LOWER_TRISOLVE_INCLUDE
