// $COPYRIGHT$

#ifndef MTL_RECURSION_UTILITIES_INCLUDE
#define MTL_RECURSION_UTILITIES_INCLUDE

#include <limits>

namespace mtl { namespace recursion {


// Splits a number into a next-smallest power of 2 and rest
std::size_t inline first_part(std::size_t n)
{
    if (n == 0) return 0;

    std::size_t  i= std::numeric_limits<std::size_t>::max()/2 + 1;

    while(i >= n) i>>= 1;
    return i;
}


// The remainder of first part
std::size_t inline second_part(std::size_t n)
{
    return n - first_part(n);
}


template <typename Matrix>
std::size_t outer_bound(Matrix const& matrix)
{
  std::size_t max_dim=std::max((matrix.num_rows(), matrix.num_cols())), bound= 1;
  for (; bound < max_dim;) bound<<= 1;
  return bound;
}



}} // namespace mtl::recursion

#endif // MTL_RECURSION_UTILITIES_INCLUDE
