// $COPYRIGHT$

#ifndef META_MATH_POWER_OF_2_INCLUDE
#define META_MATH_POWER_OF_2_INCLUDE

namespace meta_math {


// Computes the n-th power of 2
// So simple, everybody could do it, it is only there for the sake of completeness
template <unsigned long N>
struct power_of_2
{
    static const unsigned long value= X << N;
};

} // namespace meta_math

#endif // META_MATH_POWER_OF_2_INCLUDE
