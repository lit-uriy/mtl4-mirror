// $COPYRIGHT$

#ifndef MTL_BASE_TYPES_INCLUDE
#define MTL_BASE_TYPES_INCLUDE

#include <boost/type_traits.hpp>
#include <boost/numeric/mtl/tag.hpp>
#include <boost/numeric/mtl/traits.hpp>

namespace mtl {

// Types for orientation
struct row_major {};
struct col_major {};
struct morton_major {};

// Orientation type for transposed matrix
template <class T> struct transposed_orientation {};

template<> struct transposed_orientation<row_major> 
{
    typedef col_major type; 
};

template<> struct transposed_orientation<col_major> 
{
    typedef row_major type; 
};

} // namespace mtl

#endif // MTL_BASE_TYPES_INCLUDE




