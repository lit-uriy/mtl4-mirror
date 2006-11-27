// $COPYRIGHT$

#ifndef META_MATH_IS_POWER_OF_2_INCLUDE
#define META_MATH_IS_POWER_OF_2_INCLUDE

#include <boost/numeric/meta_math/least_significant_one_bit.hpp>

namespace meta_math {

template <unsigned long X>
struct is_power_of_2
{
    static const bool value= X == least_significant_one_bit<X>::value;
};

} // namespace meta_math

#endif // META_MATH_IS_POWER_OF_2_INCLUDE
