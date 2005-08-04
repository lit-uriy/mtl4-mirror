// $COPYRIGHT$

#ifndef MTL_BASE_TYPES_INCLUDE
#define MTL_BASE_TYPES_INCLUDE

#include <boost/type_traits.hpp>
// #include <boost/mpl/if.hpp>

namespace mtl {

  // typetrait to check if certain type is MTL type
  // then it provides additional types and variables
  template <class T> struct is_mtl_type 
  { 
    static bool const value= false; 
  };

  // Types for orientation
  struct row_major {};
  struct col_major {};
  struct dia_major {};

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
  // is dia_major its own transposed_orientation ???
  template<> struct transposed_orientation<dia_major> 
  {
    typedef dia_major type; 
  };

  // tag for any MTL matrix
  struct matrix_tag {};

  // Tag for any dense MTL matrix
  struct dense_tag : public matrix_tag {};

  // Tag for any sparse MTL matrix
  struct sparse_tag : public matrix_tag {};

  // Tags for dispatching on matrix types without dealing 
  // with template parameters
  struct dense2D_tag : public dense_tag {};
  struct fractalu_tag : public dense_tag {};



  // Get matrix tag for dispatching
  // Has to be specialized for each matrix
  template <class Matrix> struct which_matrix {};

} // namespace mtl

#endif // MTL_BASE_TYPES_INCLUDE





// = old code to throw away if new code works

//   // Types for indexing
//   struct c_index {};
//   struct f_index {};

//   namespace detail {
//     // non-mtl types are by default c indexed and mtl types have a declaration in 'ind'
//     template <class T, bool B> struct indexing { typedef c_index type; };
//     template <class T> struct indexing<T, true> { typedef typename T::ind type; };
//   } 

//   // Type of indexing: if mtl type it knows, otherwise c indexing
//   template <class T> struct indexing { 
//     typedef typename detail::indexing<T, is_mtl_type<T>::value>::type type; };

//   // Types are not supposed to be fortran indexed unless stated otherwise
//   template <class T> struct is_fortran_indexed { 
//     static const bool value= boost::is_same<typename indexing<T>::type, f_index>::value; };

