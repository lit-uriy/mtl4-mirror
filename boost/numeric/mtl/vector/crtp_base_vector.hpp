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

#ifndef MTL_CRTP_BASE_VECTOR_INCLUDE
#define MTL_CRTP_BASE_VECTOR_INCLUDE

#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/vector/all_vec_expr.hpp>
#include <boost/numeric/mtl/vector/assigner.hpp>
#include <boost/numeric/mtl/operation/mat_cvec_times_expr.hpp>
#include <boost/numeric/mtl/operation/mult.hpp>
#include <boost/numeric/mtl/operation/mat_vec_mult.hpp>
#include <boost/numeric/mtl/operation/mult_assign_mode.hpp>
#include <boost/numeric/mtl/operation/right_scale_inplace.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/is_distributed.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>

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

    template <typename Vector, typename Source>
    struct assign_assigner
    {
	typedef const Vector& type;
	type operator()(Vector& vector, const Source& src)
	{
	    src.assign_to(vector);
	    return vector;
	}
    };

} // namespace detail

template <typename Vector, typename Source>
struct crtp_assign 
  : boost::mpl::if_
     <boost::is_base_of<assigner_base, Source>,
      detail::assign_assigner <Vector, Source>, 
      detail::crtp_assign<Vector, Source, typename ashape::ashape<Vector>::type, typename ashape::ashape<Source>::type>
     >::type
{};

/// Assign matrix vector product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_assign<Vector, mat_cvec_times_expr<E1, E2> >
{
    typedef Vector& type;
    type operator()(Vector& vector, const mat_cvec_times_expr<E1, E2>& src)
    {
	vector.checked_change_resource(src);
	mult(src.first, src.second, vector);
	// mat_cvec_mult(src.first, src.second, vector, assign::assign_sum(), typename traits::category<E1>::type());
	return vector;
    }
};


/// Assign vector matrix product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_assign<Vector, rvec_mat_times_expr<E1, E2> >
{
    typedef Vector& type;
    type operator()(Vector& vector, const rvec_mat_times_expr<E1, E2>& src)
    {
	vector.checked_change_resource(src);
	gen_mult(src.first, src.second, vector, assign::assign_sum(), 
		 tag::row_vector(), tag::matrix(), tag::row_vector());
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
	// gen_mult(src.first, src.second, vector, assign::plus_sum(), tag::matrix(), tag::col_vector(), tag::col_vector());
	mat_cvec_mult(src.first, src.second, vector, assign::plus_sum(), typename traits::category<E1>::type());
	return vector;
    }
};

/// Assign-add vector matrix product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_plus_assign<Vector, rvec_mat_times_expr<E1, E2> >
{
    typedef Vector& type;
    type operator()(Vector& vector, const rvec_mat_times_expr<E1, E2>& src)
    {
	gen_mult(src.first, src.second, vector, assign::plus_sum(), 
		 tag::row_vector(), tag::matrix(), tag::row_vector());
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

/// Assign-subtract matrix vector product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_minus_assign<Vector, mat_cvec_times_expr<E1, E2> >
{
    typedef Vector& type;
    type operator()(Vector& vector, const mat_cvec_times_expr<E1, E2>& src)
    {
	gen_mult(src.first, src.second, vector,
		 assign::minus_sum(), tag::matrix(), tag::col_vector(), tag::col_vector());
	return vector;
    }
};

/// Assign-subtract vector matrix product by calling mult
/** Note that this does not work for arbitrary expressions. **/
template <typename Vector, typename E1, typename E2>
struct crtp_minus_assign<Vector, rvec_mat_times_expr<E1, E2> >
{
    typedef Vector& type;
    type operator()(Vector& vector, const rvec_mat_times_expr<E1, E2>& src)
    {
	gen_mult(src.first, src.second, vector, assign::minus_sum(), 
		 tag::row_vector(), tag::matrix(), tag::row_vector());
	return vector;
    }
};


/// Base class to provide vector assignment operators generically 
template <typename Vector, typename ValueType, typename SizeType>
struct crtp_vector_assign
{
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

    /// Check whether source and target have compatible resources and adapt empty target
    /** For expressions like u= v + w, u can be set to the size of v and w if still is 0. **/
    template <typename Src>
    void checked_change_resource(const Src& src) 
    {	checked_change_resource_aux(src, typename mtl::traits::is_distributed<Vector>::type()); }    

    template <typename Src>
    void checked_change_resource_aux(const Src& src, boost::mpl::false_) 
    {   checked_change_dim(mtl::vector::size(src));  }


    /// Check whether vector size is compatible or if vector is 0 change it s.
    void checked_change_dim(SizeType s)
    {
	Vector& vector= static_cast<Vector&>(*this);
	vector.check_dim(s);
	vector.change_dim(s);
    }
};


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
