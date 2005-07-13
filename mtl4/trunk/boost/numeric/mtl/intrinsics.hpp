// $COPYRIGHT$

#ifndef MTL_INTRINSICS_INCLUDE
#define MTL_INTRINSICS_INCLUDE

namespace mtl {

  // increment index if fortran and don't if c
  template <class T> 
  inline T iinc(T i, c_index) { return i; }
  template <class T> 
  inline T iinc(T i, f_index) { return i + 1; }

  // decrement index if fortran and don't if c
  template <class T> 
  inline T idec(T i, c_index) { return i; }
  template <class T> 
  inline T idec(T i, f_index) { return i - 1; }

} // namespace mtl

#endif // MTL_INTRINSICS_INCLUDE
