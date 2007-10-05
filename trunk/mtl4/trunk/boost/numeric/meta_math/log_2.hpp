// $COPYRIGHT$

#ifndef META_MATH_LOG_2_INCLUDE
#define META_MATH_LOG_2_INCLUDE

#include <boost/numeric/meta_math/is_power_of_2.hpp>

namespace meta_math {

// Computes the logarithm to the basis 2
// Without testing if power of 2 it rounds values down to next integer
template <unsigned long X>
struct log_2
{
    // BOOST_STATIC_ASSERT(is_power_of_2_meta<X>::value);
    static const unsigned long tmp= X >> 1, value= log_2<tmp>::value + 1;
};

template <> struct log_2<1>
{
    static const unsigned long value= 0;
};

template <> struct log_2<0>
{
  // #error "Logarithm of 0 is undefined"
  BOOST_STATIC_ASSERT(true); // Logarithm of 0 is undefined
};


} // namespace meta_math

#endif // META_MATH_LOG_2_INCLUDE
