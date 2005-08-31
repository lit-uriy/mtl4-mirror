// $COPYRIGHT$

#ifndef MTL_INDEX_INCLUDE
#define MTL_INDEX_INCLUDE

#include <boost/mpl/if.hpp>
#include <boost/numeric/mtl/base_types.hpp>

namespace mtl { namespace index {

// Index like in C (identical with internal representation)
struct c_index {};

// Index like Fortran
struct f_index {};

// Which index has type T
template <class T> struct which_index
{
    typedef typename boost::mpl::if_c<
          is_mtl_type<T>::value
        , typename T::index_type   // mtl data shall know their type
        , c_index                  // others are by default c
        >::type type;
};

// Change from internal representation to requested index type
template <class T> inline T change_to(c_index, T i) 
{
    return i; 
}

template <class T> inline T change_to(f_index, T i) 
{ 
    return i + 1; 
}

// Change from requested index type to internal representation
template <class T> inline T change_from(c_index, T i) 
{ 
    return i; 
}

template <class T> inline T change_from(f_index, T i) 
{ 
    return i - 1; 
}

}} // namespace mtl::index

#endif // MTL_INDEX_INCLUDE
