// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_CRTP_BASE_VECTOR_INCLUDE
#define MTL_CRTP_BASE_VECTOR_INCLUDE

#include <boost/utility/enable_if.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/vector/all_vec_expr.hpp>
#include <boost/numeric/mtl/operation/mat_cvec_times_expr.hpp>
#include <boost/numeric/mtl/operation/mult.hpp>
#include <boost/numeric/mtl/operation/mult_assign_mode.hpp>
#include <boost/numeric/mtl/operation/right_scale_inplace.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>

namespace mtl { namespace vector {


/// Base class to provide vector assignment operators generically 
template <typename Vector, typename ValueType, typename SizeType>
struct crtp_vector_assign
{
#ifdef _MSC_VER
    // MSVC has trouble with alleged ambiguity
    vec_scal_asgn_expr<Vector, ValueType>
    operator=(const ValueType& value)
    {
	return vec_scal_asgn_expr<Vector, ValueType>( static_cast<Vector&>(*this), value );
    }
#else
    /// Assign scalar to a vecttor by setting all elements to it
    template <typename Value>
    typename boost::enable_if<typename boost::is_same<typename ashape::ashape<Value>::type,
						      ashape::scal>,
			      vec_scal_asgn_expr<Vector, Value> 
                             >::type
    operator=(const Value& value)
    {
	return vec_scal_asgn_expr<Vector, Value>( static_cast<Vector&>(*this), value );
    }
#endif

    /// Assign vector expression
    template <class E>
    vec_vec_asgn_expr<Vector, E> operator=( vec_expr<E> const& e )
    {
	static_cast<Vector*>(this)->check_consistent_shape(e);
	return vec_vec_asgn_expr<Vector, E>( static_cast<Vector&>(*this), 
					     static_cast<const E&>(e) );
    }

    /// Assign matrix vector product by calling mult
    /** Note that this does not work for arbitrary expressions. **/
    template <typename E1, typename E2>
    Vector& operator=(const mat_cvec_times_expr<E1, E2>& src)
    {
	mult(src.first, src.second, static_cast<Vector&>(*this));

	return static_cast<Vector&>(*this);
    }

    /// Assign-add vector expression
    template <class E>
    vec_vec_plus_asgn_expr<Vector, E> operator+=( vec_expr<E> const& e )
    {
	static_cast<Vector*>(this)->check_consistent_shape(e);
	return vec_vec_plus_asgn_expr<Vector, E>( static_cast<Vector&>(*this), static_cast<const E&>(e) );
    }

    /// Assign-add matrix vector product by calling mult
    /** Note that this does not work for arbitrary expressions. **/
    template <typename E1, typename E2>
    Vector& operator+=(const mat_cvec_times_expr<E1, E2>& src)
    {
	gen_mult(src.first, src.second, static_cast<Vector&>(*this),
		 assign::plus_sum(), tag::matrix(), tag::vector(), tag::vector());

	return static_cast<Vector&>(*this);
    }

    /// Assign-subtract vector expression
    template <class E>
    vec_vec_minus_asgn_expr<Vector, E> operator-=( vec_expr<E> const& e )
    {
	static_cast<Vector*>(this)->check_consistent_shape(e);
	return vec_vec_minus_asgn_expr<Vector, E>( static_cast<Vector&>(*this), static_cast<const E&>(e) );
    }

    /// Assign-subtract matrix vector product by calling mult
    /** Note that this does not work for arbitrary expressions. **/
    template <typename E1, typename E2>
    Vector& operator-=(const mat_cvec_times_expr<E1, E2>& src)
    {
	gen_mult(src.first, src.second, static_cast<Vector&>(*this),
		 assign::minus_sum(), tag::matrix(), tag::vector(), tag::vector());

	return static_cast<Vector&>(*this);
    }

    /// Scale vector (in place) with scalar value 
    /** In the future, row vectors be possibly scaled by a matrix **/
    template <typename Factor>
    vec_scal_times_asgn_expr<Vector, Factor> operator*=(const Factor& alpha)
    {
	return vec_scal_times_asgn_expr<Vector, Factor>( static_cast<Vector&>(*this), alpha );
    }	

    /// Devide vector (in place) by a scalar value
	// added by Hui Li 12/11/2007
    template <typename Factor>
    vec_scal_div_asgn_expr<Vector, Factor> operator/=(const Factor& alpha)
    {
		return vec_scal_div_asgn_expr<Vector, Factor>( static_cast<Vector&>(*this), alpha );
    }	
};



template <typename Vector, typename ValueType, typename SizeType>
struct crtp_base_vector 
    : public crtp_vector_assign<Vector, ValueType, SizeType>
{};


}} // namespace mtl::vector

#endif // MTL_CRTP_BASE_VECTOR_INCLUDE
