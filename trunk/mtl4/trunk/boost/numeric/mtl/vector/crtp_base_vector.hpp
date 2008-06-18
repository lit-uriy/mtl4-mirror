// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
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

#include <boost/numeric/itl/itl_fwd.hpp>

namespace mtl { namespace vector {


namespace detail {

	template <typename Vector, typename Source, typename SCat, typename VCat>
	struct crtp_assign {};	

    /// Assign scalar to a vector by setting all values to the scalar
	template <typename Vector, typename Source, typename VCat>
	struct crtp_assign<Vector, Source, VCat, ashape::scal>
	{
		typedef vec_scal_asgn_expr<Vector, Source> type;
		type operator()(Vector& vector, const Source& src)
		{
			return type(vector, src);
		}
	};	

    /// Assign vector to a vector
	template <typename Vector, typename Source, typename Cat>
	struct crtp_assign<Vector, Source, Cat, Cat>
	{
		typedef vec_vec_asgn_expr<Vector, Source> type;
		type operator()(Vector& vector, const Source& src)
		{
			return type(vector, src);
		}
	};	
} // namespace detail

template <typename Vector, typename Source>
struct crtp_assign 
	: public detail::crtp_assign<Vector, Source, typename ashape::ashape<Vector>::type,
	                             typename ashape::ashape<Source>::type>
{};

/// Assign matrix vector product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_assign<Vector, mat_cvec_times_expr<E1, E2> >
{
	typedef Vector& type;
	type operator()(Vector& vector, const mat_cvec_times_expr<E1, E2>& src)
    {
		mult(src.first, src.second, vector);
		return vector;
	}
};

/// Assign c-style 1D-array, because it's easier to initialize.
template <typename Vector, typename Value, unsigned Rows>
struct crtp_assign<Vector, Value[Rows]>
{
	typedef Vector& type;
    type operator()(Vector& vector, const Value src[Rows])
    {
		typedef typename Collection<Vector>::size_type size_type;

		vector.checked_change_dim(Rows);

		for (size_type r= 0; r < Rows; ++r)
			vector[r]= src[r];
		return vector;
	}
};


namespace detail {

	template <typename Vector, typename Source, typename SCat, typename VCat>
	struct crtp_plus_assign {};	

    /// Assign-add vector to a vector
	template <typename Vector, typename Source, typename Cat>
	struct crtp_plus_assign<Vector, Source, Cat, Cat>
	{
		typedef vec_vec_plus_asgn_expr<Vector, Source> type;
		type operator()(Vector& vector, const Source& src)
		{
			return type( vector, src );
		}
	};	
} // namespace detail

template <typename Vector, typename Source>
struct crtp_plus_assign 
	: public detail::crtp_plus_assign<Vector, Source, typename ashape::ashape<Vector>::type, 
	                                  typename ashape::ashape<Source>::type>
{};

/// Assign-add matrix vector product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_plus_assign<Vector, mat_cvec_times_expr<E1, E2> >
{
	typedef Vector& type;
	type operator()(Vector& vector, const mat_cvec_times_expr<E1, E2>& src)
    {
		gen_mult(src.first, src.second, vector,
		         assign::plus_sum(), tag::matrix(), tag::vector(), tag::vector());
		return vector;
	}
};




namespace detail {

	template <typename Vector, typename Source, typename VCat, typename SCat>
	struct crtp_minus_assign {};	

    /// Assign-add vector to a vector
	template <typename Vector, typename Source, typename Cat>
	struct crtp_minus_assign<Vector, Source, Cat, Cat>
	{
		typedef vec_vec_minus_asgn_expr<Vector, Source> type;
		type operator()(Vector& vector, const Source& src)
		{
			return type(vector, src);
		}
	};	
} // namespace detail

template <typename Vector, typename Source>
struct crtp_minus_assign 
	: public detail::crtp_minus_assign<Vector, Source, typename ashape::ashape<Vector>::type,
	                                  typename ashape::ashape<Source>::type>
{};

/// Assign-add matrix vector product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_minus_assign<Vector, mat_cvec_times_expr<E1, E2> >
{
	typedef Vector& type;
	type operator()(Vector& vector, const mat_cvec_times_expr<E1, E2>& src)
    {
		gen_mult(src.first, src.second, vector,
		         assign::minus_sum(), tag::matrix(), tag::vector(), tag::vector());
		return vector;
	}
};





/// Base class to provide vector assignment operators generically 
template <typename Vector, typename ValueType, typename SizeType>
struct crtp_vector_assign
{
#if 0 // to be deleted
#ifdef _MSC_VER
    // MSVC has trouble with alleged ambiguity
    vec_scal_asgn_expr<Vector, ValueType>
    operator=(const ValueType& value)
    {
	return vec_scal_asgn_expr<Vector, ValueType>( static_cast<Vector&>(*this), value );
    }
#else
    /// Assign scalar to a vector by setting all elements to it
    template <typename Value>
    typename boost::enable_if<typename boost::is_same<,
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
	// std::cerr << "operator= size static_cast<const E&>(e) " << static_cast<const E&>(e).size() << "\n"; 

	static_cast<Vector*>(this)->check_consistent_shape(e);
	return vec_vec_asgn_expr<Vector, E>( static_cast<Vector&>(*this), 
					     static_cast<const E&>(e) );
    }

    /// Assign a solver_proxy, i.e. perform y= inv(A)*x for expressions y= A.solve(x)
    template<typename Solver, typename VectorIn, bool adjoint>
    Vector& operator=(const itl::solver_proxy<Solver, VectorIn, adjoint>& proxy)
    {
	proxy.solve(static_cast<Vector&>(*this));
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
#endif
    /// Templated assignment implemented by functor to allow for partial specialization
    template <typename E>
    typename boost::disable_if<boost::is_same<Vector, E>, 
		                       typename crtp_assign<Vector, E>::type>::type
	operator=(const E& e)
    {
		return crtp_assign<Vector, E>()(static_cast<Vector&>(*this), e);
    }

    /// Assign-add vector expression
    template <class E>
	typename crtp_plus_assign<Vector, E>::type operator+=(const E& e)
    {
		return crtp_plus_assign<Vector, E>()(static_cast<Vector&>(*this), e);
    }

    /// Assign-subtract vector expression
    template <class E>
	typename crtp_minus_assign<Vector, E>::type operator-=(const E& e)
    {
		return crtp_minus_assign<Vector, E>()(static_cast<Vector&>(*this), e);
    }

#if 0
    /// Assign-subtract matrix vector product by calling mult
    /** Note that this does not work for arbitrary expressions. **/
    template <typename E1, typename E2>
    Vector& operator-=(const mat_cvec_times_expr<E1, E2>& src)
    {
	gen_mult(src.first, src.second, static_cast<Vector&>(*this),
		 assign::minus_sum(), tag::matrix(), tag::vector(), tag::vector());

	return static_cast<Vector&>(*this);
    }
#endif
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

    /// Check whether vector size is compatible or if vector is 0 change it s.
    void checked_change_dim(SizeType s)
    {
		Vector& vector= static_cast<Vector&>(*this);
		vector.check_dim(s);
		vector.change_dim(s);
    }
};



#if 0
/// Base class to provide vector assignment operators generically 
template <typename Vector, typename ValueType, typename SizeType>
struct crtp_vector_assign
{
    /// Check whether vector sizes are compatible or if vector is 0 by 0 change it to r by c.
    void checked_change_dim(SizeType s)
    {
		Vector& vector= static_cast<Vector&>(*this);
		vector.check_dim(s);
		vector.change_dim(s);
    }

    /// Templated assignment implemented by functor to allow for partial specialization
    // Despite there is only an untemplated assignement and despite the disable_if MSVC whines about ambiguity :-!
    template <typename Source>
    typename boost::disable_if<typename boost::is_same<Vector, Source>, Vector&>::type
    operator=(const Source& src)
    {
		return crtp_assign<Source, Vector>()(src, static_cast<Vector&>(*this));
    }

    template <typename Source>
    Vector& operator+=(const Source& src)
    {
		return crtp_plus_assign<Source, Vector>()(src, static_cast<Vector&>(*this));
    }
    
    template <typename Source>
    Vector& operator-=(const Source& src)
    {
		return crtp_minus_assign<Source, Vector>()(src, static_cast<Vector&>(*this));
    }
    
    /// Scale vector (in place) with scalar value or other vector
    template <typename Factor>
    Vector& operator*=(const Factor& alpha)
    {
		right_scale_inplace(static_cast<Vector&>(*this), alpha);
		return static_cast<Vector&>(*this);
    }
    
    /// Divide vector (in place) by scalar value
    // added by Hui Li
    template <typename Factor>
    Vector& operator/=(const Factor& alpha)
    {
		divide_by_inplace(static_cast<Vector&>(*this), alpha);
		return static_cast<Vector&>(*this);
    }
};
#endif



template <typename Vector, typename ValueType, typename SizeType>
struct const_crtp_base_vector 
{};

template <typename Vector, typename ValueType, typename SizeType>
struct mutable_crtp_base_vector 
	: public crtp_vector_assign<Vector, ValueType, SizeType>
{};



template <typename Vector, typename ValueType, typename SizeType>
struct crtp_base_vector 
    : boost::mpl::if_<boost::is_const<Vector>,
	                  const_crtp_base_vector<Vector, ValueType, SizeType>,
		              mutable_crtp_base_vector<Vector, ValueType, SizeType>
                     >::type
{};


}} // namespace mtl::vector

#endif // MTL_CRTP_BASE_VECTOR_INCLUDE
