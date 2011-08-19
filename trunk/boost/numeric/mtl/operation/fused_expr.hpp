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
#include <boost/mpl/and.hpp>
#include <boost/numeric/mtl/operation/index_evaluator.hpp>
#include <boost/numeric/mtl/utility/index_evaluatable.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>

namespace mtl {

/// Expression template for fusing other expression 
template <typename T, typename U>
struct fused_expr
{
    typedef boost::mpl::and_<traits::forward_index_evaluatable<T>, traits::forward_index_evaluatable<U> >   forward;
    typedef boost::mpl::and_<traits::backward_index_evaluatable<T>, traits::backward_index_evaluatable<U> > backward;

    fused_expr(T& first, U& second) : first(first), second(second) {}
 
    ~fused_expr() { eval(forward(), backward()); }

    template <typename TT, typename UU>
    void forward_eval_loop(TT first_eval, UU second_eval, boost::mpl::false_)
    {	
	vampir_trace<3047> tracer;
	MTL_DEBUG_THROW_IF(mtl::vector::size(first_eval) != mtl::vector::size(second_eval), incompatible_size());	

	for (std::size_t i= 0, s= size(first_eval); i < s; i++) {
	    first_eval(i); second_eval(i);
	}	
    }

    template <typename TT, typename UU>
    void forward_eval_loop(const TT& const_first_eval, const UU& const_second_eval, boost::mpl::true_)
    {	
	vampir_trace<3048> tracer;
	// hope there is a more elegant way; copying the arguments causes errors due to double destructor evaluation
	TT& first_eval= const_cast<TT&>(const_first_eval);  
	UU& second_eval= const_cast<UU&>(const_second_eval);
	MTL_DEBUG_THROW_IF(mtl::vector::size(first_eval) != mtl::vector::size(second_eval), incompatible_size());	

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
    }

    // Forward evaluation dominates backward
    template <bool B2>
    void eval(boost::mpl::true_, boost::mpl::bool_<B2>)
    {
#ifdef MTL_LAZY_LOOP_WO_UNROLL
	typedef boost::mpl::false_                                                                              to_unroll;
#else
	typedef boost::mpl::and_<traits::unrolled_index_evaluatable<T>, traits::unrolled_index_evaluatable<U> > to_unroll;
#endif
	// Currently lazy evaluation is only available on vector expressions, might change in the future
	// std::cout << "Forward evaluation\n";
	forward_eval_loop(index_evaluator(first), index_evaluator(second), to_unroll()); 
    }

    // Sequential evaluation
    template <bool B1, bool B2>
    void eval(boost::mpl::false_, boost::mpl::false_)
    { evaluate_lazy(first); evaluate_lazy(second); }

    T& first;
    U& second;
};


} // namespace mtl

#endif // MTL_FUSED_EXPR_INCLUDE
