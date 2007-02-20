// $COPYRIGHT$

#ifndef META_MATH_MAX_INCLUDE
#define META_MATH_MAX_INCLUDE

namespace meta_math {

template <long int x, long int y>
struct max
{
    typedef long int type;
    static long int const value = x < y ? y : x;
};

} // namespace meta_math

#endif // META_MATH_MAX_INCLUDE
