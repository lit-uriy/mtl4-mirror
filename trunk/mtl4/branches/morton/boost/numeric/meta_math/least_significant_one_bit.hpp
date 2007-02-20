// $COPYRIGHT$

#ifndef META_MATH_LEAST_SIGNIFICANT_ONE_BIT_INCLUDE
#define META_MATH_LEAST_SIGNIFICANT_ONE_BIT_INCLUDE

namespace meta_math {

template <unsigned long X>
struct least_significant_one_bit
{
  static const unsigned long value= (X ^ X-1) + 1 >> 1;
};


} // namespace meta_math

#endif // META_MATH_LEAST_SIGNIFICANT_ONE_BIT_INCLUDE
