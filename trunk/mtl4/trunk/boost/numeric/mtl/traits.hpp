// $COPYRIGHT$

#ifndef MTL_TRAITS_INCLUDE
#define MTL_TRAITS_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>

namespace mtl { namespace traits {

// Get tag for dispatching matrices, vectors, ...
// Has to be specialized for each matrix, vector, ...
template <class Collection> struct category 
{
    typedef tag::unknown type;
};

}} // namespace mtl::traits 

#endif // MTL_TRAITS_INCLUDE
