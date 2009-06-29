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

#ifndef MTL_DISTRIBUTION_INCLUDE
#define MTL_DISTRIBUTION_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/compute_summand.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>

namespace mtl {

namespace vector {



    template <typename EE1, typename EE2, typename SSFunctor> 
    typename DistributedVector< vec_vec_aop_expr<EE1, EE2, SSFunctor> >::distribution_type
    inline distribution(const vec_vec_aop_expr<EE1, EE2, SSFunctor>& expr)
    {
	MTL_DEBUG_THROW_IF(distribution(expr.first) != distribution(expr.second), incompatible_distribution());
	return distribution(expr.first);
    }

    template <typename EE1, typename EE2, typename SSFunctor> 
    typename DistributedVector< vector::vec_vec_pmop_expr<EE1, EE2, SSFunctor> >::distribution_type
    inline distribution(const vec_vec_pmop_expr<EE1, EE2, SSFunctor>& expr)
    {
	MTL_DEBUG_THROW_IF(distribution(expr.first.value) != distribution(expr.second.value), incompatible_distribution());
	return distribution(expr.first.value);
    }

    template <typename F, typename C> 
    typename DistributedVector< map_view<F, C> >::distribution_type
    inline distribution(const map_view<F, C>& expr)
    {
	return distribution(expr.ref);
    }

} // vector

namespace operation {

    template <typename Expr>
    typename DistributedVector< typename compute_summand<Expr>::type >::distribution_type
    inline distribution(const compute_summand<Expr>& s)
    {
	return distribution(s.value);
    }

}

} // namespace mtl

#endif // MTL_DISTRIBUTION_INCLUDE
