// $COPYRIGHT$

#ifndef MTL_INTRINSICS_INCLUDE
#define MTL_INTRINSICS_INCLUDE

namespace mtl {

//   // increment index if fortran and don't if c
//   // no overhead for c indexing
//   template <class T> 
//   inline T iinc(T i, c_index) { return i; }
//   template <class T> 
//   inline T iinc(T i, f_index) { return i + 1; }

//   // decrement index if fortran and don't if c
//   // no overhead for c indexing
//   template <class T> 
//   inline T idec(T i, c_index) { return i; }
//   template <class T> 
//   inline T idec(T i, f_index) { return i - 1; }

//   // increment index if fortran and second isn't
//   // vice versa if second is fortran and first isn't decrement
//   // no overhead for equal indexing
//   template <class T> 
//   inline T iinc_wrt(T i, c_index, c_index) { return i; }
//   template <class T> 
//   inline T iinc_wrt(T i, f_index, f_index) { return i; }
//   template <class T> 
//   inline T iinc_wrt(T i, f_index, c_index) { return i + 1; }
//   template <class T> 
//   inline T iinc_wrt(T i, c_index, f_index) { return i - 1; }
  
//   // correspondingly for decreasing, opposite of iinc_wrt
//   template <class T, class Index1, class Index2> 
//   inline T idec_wrt(T i, Index1 ind1, Index2 ind2) { 
//     return iinc_wrt(i, ind2, ind1); }
  

} // namespace mtl

#endif // MTL_INTRINSICS_INCLUDE
