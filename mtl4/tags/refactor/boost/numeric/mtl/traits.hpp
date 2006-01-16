// $COPYRIGHT$

#ifndef MTL_TRAITS_INCLUDE
#define MTL_TRAITS_INCLUDE

#include <boost/numeric/mtl/tag.hpp>

namespace mtl { namespace traits {

// typetrait to check if certain type is MTL type
// then it provides additional types and variables
template <class T> struct is_mtl_type 
{ 
    static bool const value= false; 
};

// Get matrix tag for dispatching
// Has to be specialized for each matrix
template <class Matrix> struct matrix_category 
{
    typedef tag::unknown type;
};

}} // namespace mtl::traits 

#endif // MTL_TRAITS_INCLUDE
