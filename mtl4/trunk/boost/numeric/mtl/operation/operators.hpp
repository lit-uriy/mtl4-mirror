// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_OPERATORS_INCLUDE
#define MTL_OPERATORS_INCLUDE

#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/matrix/operators.hpp>
//#include <boost/numeric/mtl/vector/operators.hpp>
#include <boost/numeric/mtl/operation/mult_result.hpp>
#include <boost/numeric/mtl/matrix/all_mat_expr.hpp>


namespace mtl {

/// Multiplication for all supported types of operations
/** Enable-if-like technique make sure that only called when properly defined **/
template <typename Op1, typename Op2>
typename traits::mult_result<Op1, Op2>::type
inline operator*(const Op1& op1, const Op2& op2)
{
    return typename traits::mult_result<Op1, Op2>::type(op1, op2);
}



#if 0
// If we implement operator* in a different way for each type of operation it would like the following

// Scale matrix from left
template <typename Op1, typename Op2>
typename traits::mult_result_if_equal<Op1, Op2, ashape::scal_mat_mult>::type
inline operator*(const Op1& op1, const Op2& op2)
{
    return typename traits::mult_result_if_equal<Op1, Op2, ashape::scal_mat_mult>::type(op1, op2);
}
#endif

} // namespace mtl

#endif // MTL_OPERATORS_INCLUDE
