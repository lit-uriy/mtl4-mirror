// $COPYRIGHT$

#ifndef MTL_BASE_TYPES_INCLUDE
#define MTL_BASE_TYPES_INCLUDE

#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

namespace mtl {

  // typetrait to check if certain type is MTL type
  // then it provides additional types and variables
  template <class T> struct is_mtl_type { static const bool value= false; };

  // Types for orientation
  struct row_major {};
  struct col_major {};
  struct dia_major {};

  template <class T> struct transposed_orientation {};
  template<> struct transposed_orientation<row_major> {
    typedef col_major type; };
  template<> struct transposed_orientation<col_major> {
    typedef row_major type; };
  // is dia_major its own transposed_orientation ???

  // Types for indexing
  struct c_index {};
  struct f_index {};

  // Types are not supposed to be fortran indexed unless stated otherwise
  template <class T> struct is_fortran_indexed { 
    static const bool value= boost::is_same<typename indexing<T>::type, c_index>::value; };

  // Type of indexing: if mtl type it knows, otherwise c indexing
  template <class T> struct indexing { 
    typedef typename if_c<is_mtl_type<T>::value, typename T::ind, c_index>::type type; };
  
} // namespace mtl

#endif // MTL_BASE_TYPES_INCLUDE
