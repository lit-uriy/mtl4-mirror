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

#ifndef MTL_OPERATION_DISTRIBUTION_INCLUDE
#define MTL_OPERATION_DISTRIBUTION_INCLUDE

#include <boost/type_traits/add_reference.hpp>
#include <boost/numeric/mtl/utility/distribution.hpp>
#include <boost/numeric/mtl/vector/map_view.hpp>
#include <boost/numeric/mtl/vector/vec_vec_aop_expr.hpp>
#include <boost/numeric/mtl/vector/vec_vec_pmop_expr.hpp>


namespace mtl { 

    namespace vector {

	template <typename Functor, typename Vector>
	typename boost::add_reference<typename mtl::traits::distribution<map_view<Functor, Vector> >::type const>::type 
	inline distribution(const map_view<Functor, Vector>& v)
	{
	    return distribution(v.reference());
	}

	template <typename E1, typename E2, typename Functor>
	typename boost::add_reference<typename mtl::traits::distribution<vec_vec_aop_expr<E1, E2, Functor> >::type const>::type
	inline distribution(const vec_vec_aop_expr<E1, E2, Functor>& v)
	{
	    MTL_DEBUG_THROW_IF(distribution(v.first) != distribution(v.second), incompatible_distribution());
	    return distribution(v.first);
	}

	template <typename E1, typename E2, typename Functor>
	typename boost::add_reference<typename mtl::traits::distribution<vec_vec_pmop_expr<E1, E2, Functor> >::type const>::type
	inline distribution(const vec_vec_pmop_expr<E1, E2, Functor>& v)
	{
	    MTL_DEBUG_THROW_IF(distribution(v.reference_first()) != distribution(v.reference_second()), incompatible_distribution());
	    return distribution(v.reference_first());
	}

    } // namespace vector

    template <typename Matrix, typename CVector>
    typename boost::add_reference<typename mtl::traits::distribution<mtl::mat_cvec_times_expr<Matrix, CVector> >::type const>::type
    inline distribution(const mtl::mat_cvec_times_expr<Matrix, CVector>& expr)
    {
	return row_distribution(expr.first);
    }

} // namespace mtl

#endif // MTL_OPERATION_DISTRIBUTION_INCLUDE
