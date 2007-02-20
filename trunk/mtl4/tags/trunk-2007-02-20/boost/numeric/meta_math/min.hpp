// $COPYRIGHT$

#ifndef META_MATH_MIN_INCLUDE
#define META_MATH_MIN_INCLUDE

namespace meta_math {

template <long int x, long int y>
struct min
{
  typedef long int type;
  static long int const value = x < y ? x : y;
};

} // namespace meta_math

#endif // META_MATH_MIN_INCLUDE
