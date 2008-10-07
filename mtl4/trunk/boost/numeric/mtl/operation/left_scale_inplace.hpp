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

#ifndef MTL_LEFT_SCALE_INPLACE_INCLUDE
#define MTL_LEFT_SCALE_INPLACE_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/operation/assign_each_nonzero.hpp>
#include <boost/numeric/mtl/operation/mult.hpp>

#include <boost/lambda/lambda.hpp>


namespace mtl {

    namespace impl {

	template <typename Factor, typename Collection>
	inline Collection& left_scale_inplace(const Factor& alpha, tag::scalar, Collection& c)
	{
	    assign_each_nonzero(c, alpha * boost::lambda::_1);
	    return c;
	}

	template <typename Factor, typename Collection>
	inline Collection& left_scale_inplace(const Factor& alpha, tag::matrix, Collection& c)
	{
	    Collection tmp(alpha * c);
	    swap(tmp, c);
	    return c;
	}
    }
    
    namespace matrix {
	
	/// Scale matrix \p c from left with scalar or matrix factor \p alpha; \p c is altered
	template <typename Factor, typename Matrix>
	typename traits::enable_if_matrix<Matrix, Matrix&>::type
	inline left_scale_inplace(const Factor& alpha, Matrix& A)
	{
	    return mtl::impl::left_scale_inplace(alpha, typename traits::category<Factor>::type(), A);
	}
    }

    namespace vector {
	
	/// Scale vector \p c from left with scalar or matrix factor \p alpha; \p c is altered
	template <typename Factor, typename Vector>
	typename traits::enable_if_vector<Vector, Vector&>::type
	inline left_scale_inplace(const Factor& alpha, Vector& v)
	{
	    return mtl::impl::left_scale_inplace(alpha, typename traits::category<Factor>::type(), v);
	}
    }

    using vector::left_scale_inplace;
    using matrix::left_scale_inplace;


#if 0

/// Scale collection \p c from left with scalar factor \p alpha; \p c is altered
template <typename Factor, typename Collection>
void left_scale_inplace(const Factor& alpha, tag::scalar, Collection& c)
{
    assign_each_nonzero(c, alpha * boost::lambda::_1);
}

template <typename Factor, typename Matrix>
void left_scale_inplace(const Factor& alpha, tag::matrix, Matrix& m, tag::matrix)
{
    using mtl::swap;

    Matrix tmp(num_rows(m), num_cols(m));
    mult(alpha, m, tmp);
    swap(m, tmp);
}

template <typename Factor, typename Vector>
void left_scale_inplace(const Factor& alpha, tag::matrix, Vector& v, tag::vector)
{
    using mtl::swap;

    Vector tmp(size(v));
    mult(alpha, v, tmp);
    swap(v, tmp);
}

/// Scale collection \p c from left with matrix factor \p alpha; \p c is altered
template <typename Factor, typename Collection>
void left_scale_inplace(const Factor& alpha, tag::matrix, Collection& c)
{
    // Need to dispatch further to use different constructors for temporary
    left_scale_inplace(alpha, tag::matrix(), c, typename traits::category<Collection>::type());
}

/// Scale collection \p c from left with factor \p alpha; \p c is altered
template <typename Factor, typename Collection>
void left_scale_inplace(const Factor& alpha, Collection& c)
{
    // Dispatch between scalar and matrix factors
    left_scale_inplace(alpha, typename traits::category<Factor>::type(), c);
}


#endif

} // namespace mtl

#endif // MTL_LEFT_SCALE_INPLACE_INCLUDE
