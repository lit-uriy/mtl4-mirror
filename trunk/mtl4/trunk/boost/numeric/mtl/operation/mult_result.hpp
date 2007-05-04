// $COPYRIGHT$

#ifndef MTL_MULT_RESULT_INCLUDE
#define MTL_MULT_RESULT_INCLUDE

#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/matrix/map_view.hpp>

namespace mtl { namespace traits {

template <typename Op1, typename Op2, typename MultOp> struct mult_result_aux;
template <typename Op1, typename Op2, typename MultOp1, typename MultOp2> struct mult_result_if_equal_aux;

/// Result type for multiplying arguments of types Op1 and Op2
/** Can be used in enable-if-style as type is only defined when appropriate **/
template <typename Op1, typename Op2>
struct mult_result 
    : public mult_result_aux<Op1, Op2, typename ashape::mult_op<typename ashape::ashape<Op1>::type, 
								typename ashape::ashape<Op2>::type >::type>
{}; 


/// Result type for multiplying arguments of types Op1 and Op2 if operation is classified as MultOp
/** Can be used in enable-if-style as type is only defined when appropriate **/
template <typename Op1, typename Op2, typename MultOp>
struct mult_result_if_equal 
    : public mult_result_if_equal_aux<Op1, Op2, typename ashape::mult_op<typename ashape::ashape<Op1>::type, 
									 typename ashape::ashape<Op2>::type>::type,
				      MultOp>
{};

template <typename Op1, typename Op2, typename MultOp1, typename MultOp2>
struct mult_result_if_equal_aux {};

template <typename Op1, typename Op2, typename MultOp>
struct mult_result_if_equal_aux<Op1, Op2, MultOp, MultOp>
    : public mult_result_aux<Op1, Op2, typename ashape::mult_op<typename ashape::ashape<Op1>::type, 
								typename ashape::ashape<Op2>::type >::type>
{};


/// Result type for multiplying arguments of types Op1 and Op2
/** MultOp according to the algebraic shapes **/
template <typename Op1, typename Op2, typename MultOp>
struct mult_result_aux {};

/// Scale matrix from left
template <typename Op1, typename Op2>
struct mult_result_aux<Op1, Op2, ::mtl::ashape::scal_mat_mult> 
{
    typedef matrix::scaled_view<Op1, Op2> type;
};

/// Scale matrix from right
template <typename Op1, typename Op2>
struct mult_result_aux<Op1, Op2, ::mtl::ashape::mat_scal_mult> 
{
    typedef matrix::scaled_view<Op1, Op2> type;
};

#if 0
/// Multiply matrices
template <typename Op1, typename Op2>
struct mult_result_aux<Op1, Op2, ::mtl::ashape::scal_mat_mult> 
{
    typedef matrix::mat_mat_times_expr<Op1, Op2> type;
};
#endif



}} // namespace mtl::traits

#endif // MTL_MULT_RESULT_INCLUDE
