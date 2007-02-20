// $COPYRIGHT$

#ifndef META_MATH_LOOP1_INCLUDE
#define META_MATH_LOOP1_INCLUDE

// See loop3.hpp for example

namespace meta_math {

template <unsigned long Index0, unsigned long Max0>
struct loop1
{
    static unsigned long const index0= Index0 - 1, next_index0= Index0 + 1;
};


template <unsigned long Max0>
struct loop1<Max0, Max0>
{
    static unsigned long const index0= Max0 - 1;
};


} // namespace meta_math

#endif // META_MATH_LOOP1_INCLUDE
