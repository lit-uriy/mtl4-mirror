// $COPYRIGHT$

#ifndef MTL_LEFT_SCALE_INPLACE_INCLUDE
#define MTL_LEFT_SCALE_INPLACE_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/operation/assign_each_nonzero.hpp>

#include <boost/lambda/lambda.hpp>


namespace mtl {


/// Scale collection \p c from left with scalar factor \p alpha; \p c is altered
template <typename Factor, typename Collection>
void left_scale_inplace(const Factor& alpha, tag::scalar, Collection& c)
{
    using namespace boost::lambda;
    assign_each_nonzero(c, alpha * _1);
}

/// Scale collection \p c from left with factor \p alpha; \p c is altered
template <typename Factor, typename Collection>
void left_scale_inplace(const Factor& alpha, Collection& c)
{
    // Dispatch between scalar and matrix factors
    left_scale_inplace(alpha, typename traits::category<Factor>::type(), c);
}




} // namespace mtl

#endif // MTL_LEFT_SCALE_INPLACE_INCLUDE
