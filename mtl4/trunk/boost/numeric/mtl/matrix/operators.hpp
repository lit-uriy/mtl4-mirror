// $COPYRIGHT$

#ifndef MTL_OPERATORS_INCLUDE
#define MTL_OPERATORS_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/matrix/all_mat_expr.hpp>



namespace mtl { namespace matrix {

template <typename E1, typename E2>
inline mat_mat_plus_expr<E1, E2>
operator+ (const mat_expr<E1>& e1, const mat_expr<E2>& e2)
{
    // do not add matrices with inconsistent value types
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<E1>::type, 
			                typename ashape::ashape<E1>::type>::value));
    return mat_mat_plus_expr<E1, E2>(e1.ref, e2.ref);
}


#if 0
// Planned for future optimizations on sums of dense matrix expressions
template <typename E1, typename E2>
inline dmat_dmat_plus_expr<E1, E2>
operator+ (const dmat_expr<E1>& e1, const dmat_expr<E2>& e2)
{
    // do not add matrices with inconsistent value types
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<E1>::type, 
			                typename ashape::ashape<E1>::type>::value));
    return dmat_dmat_plus_expr<E1, E2>(e1.ref, e2.ref);
}
#endif


template <typename E1, typename E2>
inline mat_mat_minus_expr<E1, E2>
operator- (const mat_expr<E1>& e1, const mat_expr<E2>& e2)
{
    // do not add matrices with inconsistent value types
    BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<E1>::type, 
			                typename ashape::ashape<E1>::type>::value));
    return mat_mat_minus_expr<E1, E2>(e1.ref, e2.ref);
}



}} // namespace mtl::matrix

#endif // MTL_OPERATORS_INCLUDE
