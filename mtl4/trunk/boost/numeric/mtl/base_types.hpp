// $COPYRIGHT$

#ifndef MTL_BASE_TYPES_INCLUDE
#define MTL_BASE_TYPES_INCLUDE

namespace mtl {

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


} // namespace mtl

#endif // MTL_BASE_TYPES_INCLUDE
