// $COPYRIGHT$

#ifndef META_MATH_ABS_INCLUDE
#define META_MATH_ABS_INCLUDE

namespace meta_math {

template <long int x>
struct abs
{
  static long int const value = x < 0 ? -x : x;
};


} // namespace meta_math

#endif // META_MATH_ABS_INCLUDE
