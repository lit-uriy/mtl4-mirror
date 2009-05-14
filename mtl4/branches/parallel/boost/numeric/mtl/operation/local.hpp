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

#ifndef MTL_LOCAL_INCLUDE
#define MTL_LOCAL_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/compute_summand.hpp>

namespace mtl {

namespace vector {

    /// Return local part of assign expression
    /*  Should be only defined for distributed expressions (like enable_if) **/
    template <typename E1, typename E2, typename SFunctor> 
    typename DistributedCollection< vec_vec_aop_expr<E1, E2, SFunctor> >::local_type
    inline local(vec_vec_aop_expr<E1, E2, SFunctor>& expr)
    {
	return typename DistributedCollection< vec_vec_aop_expr<E1, E2, SFunctor> >::local_type(local(expr.first), local(expr.second));
    }

    /// Return local part of binary expression
    /*  Should be only defined for distributed expressions (like enable_if) **/
    template <typename E1, typename E2, typename SFunctor> 
    typename DistributedCollection< vec_vec_op_expr<E1, E2, SFunctor> >::local_type
    inline local(const vec_vec_op_expr<E1, E2, SFunctor>& expr)
    {
        return typename DistributedCollection< vec_vec_op_expr<E1, E2, SFunctor> >::local_type(local(expr.first), local(expr.second));
    }
        
    /// Return local part of binary expression
    /*  Should be only defined for distributed expressions (like enable_if) **/
    template <typename E1, typename E2> 
    typename DistributedCollection< vec_vec_minus_expr<E1, E2> >::local_type
    inline local(const vec_vec_minus_expr<E1, E2>& expr)
    {
        return typename DistributedCollection< vec_vec_minus_expr<E1, E2> >::local_type(local(expr.first.value), local(expr.second.value));
    }

    /// Return local part of binary expression
    /*  Should be only defined for distributed expressions (like enable_if) **/
    template <typename E1, typename E2> 
    typename DistributedCollection< vec_vec_plus_expr<E1, E2> >::local_type
    inline local(const vec_vec_plus_expr<E1, E2>& expr)
    {
	std::cout << "In local(vec_vec_plus_expr)" << std::endl;
        return typename DistributedCollection< vec_vec_plus_expr<E1, E2> >::local_type(local(expr.first.value), local(expr.second.value));
    }
         
} // namespace vector


namespace operation {

    template <typename Expr>
    typename DistributedCollection< typename compute_summand<Expr>::type >::local_type
    inline local(const compute_summand<Expr>& s)
    {
	return local(s.value);
    }

}

} // namespace mtl

#endif // MTL_LOCAL_INCLUDE

#if 0
./boost/numeric/mtl/operation/local.hpp:56: error: no matching function for call to 




'mtl::vector::vec_vec_plus_expr<
      mtl::vector::vec_vec_plus_expr<
          mtl::vector::dense_vector<
               float, mtl::vector::parameters<mtl::tag::col_major, mtl::vector::non_fixed::dimension, false, false> >, 
          mtl::vector::dense_vector<float, mtl::vector::parameters<mtl::tag::col_major, mtl::vector::non_fixed::dimension, false, false> > 
      >, 
      mtl::vector::dense_vector<
          float, mtl::vector::parameters<mtl::tag::col_major, mtl::vector::non_fixed::dimension, false, false> > 
>::vec_vec_plus_expr(
      mtl::vector::vec_vec_plus_expr<
           mtl::vector::distributed<mtl::vector::dense_vector<float, mtl::vector::parameters<mtl::tag::col_major, mtl::vector::non_fixed::dimension, false, false> >, 
                                    mtl::par::block_distribution>, 
           mtl::vector::distributed<mtl::vector::dense_vector<float, mtl::vector::parameters<mtl::tag::col_major, mtl::vector::non_fixed::dimension, false, false> >, 
                                    mtl::par::block_distribution> 
      >, 
      mtl::vector::distributed<mtl::vector::dense_vector<float, mtl::vector::parameters<mtl::tag::col_major, mtl::vector::non_fixed::dimension, false, false> >, 
                               mtl::par::block_distribution>
)'
#endif
