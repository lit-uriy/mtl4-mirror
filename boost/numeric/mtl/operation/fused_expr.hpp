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

#ifndef MTL_FUSED_EXPR_INCLUDE
#define MTL_FUSED_EXPR_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/numeric/mtl/operation/index_evaluator.hpp>
#include <boost/numeric/mtl/utility/index_evaluatable.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>

namespace mtl {

/// Expression template for fusing other expression 
template <typename T, typename U>
struct fused_expr
{
    fused_expr(T& first, U& second) : first(first), second(second) {}
 
    ~fused_expr() { eval(traits::index_evaluatable<T>(), traits::index_evaluatable<U>()); }

    template <typename TT, typename UU>
    void eval_loop(TT first_eval, UU second_eval)
    {	
	MTL_DEBUG_THROW_IF(mtl::vector::size(first_eval) != mtl::vector::size(second_eval), incompatible_size());	

#ifdef MTL_LAZY_LOOP_WO_UNROLL
	for (std::size_t i= 0, s= size(first_eval); i < s; i++) {
	    first_eval(i); second_eval(i);
	}	
#else
	const std::size_t s= size(first_eval), sb= s >> 2 << 2;

	for (std::size_t i= 0; i < sb; i+= 4) {
	    first_eval.template at<0>(i); second_eval.template at<0>(i);
	    first_eval.template at<1>(i); second_eval.template at<1>(i);
	    first_eval.template at<2>(i); second_eval.template at<2>(i);
	    first_eval.template at<3>(i); second_eval.template at<3>(i);
	}

	for (std::size_t i= sb; i < s; i++) {
	    first_eval(i); second_eval(i);
	}
#endif
    }

    void eval(boost::mpl::true_, boost::mpl::true_)
    {
	// Currently lazy evaluation is only available on vector expressions, might change in the future
	eval_loop(index_evaluator(first), index_evaluator(second)); 
    }

    template <bool B1, bool B2>
    void eval(boost::mpl::bool_<B1>, boost::mpl::bool_<B2>)
    { evaluate_lazy(first); evaluate_lazy(second); }

    T& first;
    U& second;
};


} // namespace mtl

#endif // MTL_FUSED_EXPR_INCLUDE
