// $COPYRIGHT$

#ifndef MTL_MAT_VEC_MULT_INCLUDE
#define MTL_MAT_VEC_MULT_INCLUDE

#include <mtl/property_map.hpp>
#include <mtl/intrinsics.hpp>
#include <mtl/base_types.hpp>
#include <mtl/fractalu.hpp>
#include <iostream>

namespace mtl {
  using std::size_t;

  // general matrix vector product for dense matrices (might be slow)
  // for dense2D: row major ~ 4 times faster, column major ~ 3 times slower
  template <class Matrix, class Vector_in, class Vector_out> 
  void dense_mat_vec_mult(const Matrix& ma, const Vector_in& vin, Vector_out& vout) {
    // if (ma.rows() != vout.size()) throw something;
    // if (ma.cols() != vin.size()) throw something else;
    size_t mi= 0, mrows= ma.rows(), mj_start= 0, mcols= ma.cols(), vi= 0, vj_start= 0;
    if (is_fortran_indexed<Matrix>::value) mi++, mj_start++, mrows++, mcols++;
    if (is_fortran_indexed<Vector_in>::value) vj_start++;
    if (is_fortran_indexed<Vector_out>::value) vi++;
  
#   ifdef NO_TMP_ACCUMULATE
      std::cout << "not_tmp_accumulate\n";
      for (; mi != mrows; mi++, vi++) {
	vout[vi]= (typename Vector_out::value_type) 0;
	for (size_t mj= mj_start, vj= vj_start; mj != mcols; mj++, vj++)
	  vout[vi]+= ma(mi, mj) * vin[vj]; }
#   else // use temporary in hope compiler uses register for summation
      for (; mi != mrows; mi++, vi++) {
	typename Vector_out::value_type tmp(0);
	for (size_t mj= mj_start, vj= vj_start; mj != mcols; mj++, vj++)
	  tmp+= ma(mi, mj) * vin[vj]; 
	vout[vi]= tmp;  }
#   endif
  }

  // general matrix vector product that iterates over matrix elements (might be slow)
  template <class Matrix, class Vector_in, class Vector_out> 
  void mat_vec_mult(const Matrix& ma, const Vector_in& vin, Vector_out& vout) {
    // if (ma.rows() != vout.size()) throw something;
    // if (ma.cols() != vin.size()) throw something else;
    typename indexing<Matrix>::type      mind;
    typename indexing<Vector_in>::type   viind;
    typename indexing<Vector_out>::type  voind;

    // vout= reinterpret_cast<typename Vector_out::value_type>(0);
    // ugly hack, only for testing
    for (size_t i= 0; i < vout.size(); i++)
      vout[iinc(i, voind)]= (typename Vector_out::value_type) 0;
    typename Matrix::el_cursor_type cursor= ma.ebegin(), end= ma.eend();
    for (; cursor != end; ++cursor)
      vout[idec_wrt(row(ma, *cursor), mind, voind)]+=        // corrected index
	value(ma, *cursor) * vin[idec_wrt(col(ma, *cursor), mind, viind)];
  }

  template <class ELT, size_t NF, class Vector_in, class Vector_out> 
  void mat_vec_mult(const fractalu<ELT, NF>& ma, const Vector_in& vin, Vector_out& vout) {
    // if (ma.rows() != vout.size()) throw something;
    // if (ma.cols() != vin.size()) throw something else;
    typedef fractalu<ELT, NF>            Matrix;
    typedef typename Vector_out::value_type ovalue_type;
    typedef typename Matrix::el_cursor_type el_cursor_type;
    c_index                              mind;
    typename indexing<Vector_in>::type   viind;
    typename indexing<Vector_out>::type  voind;

    // ugly hack, only for testing
    for (size_t i= 0; i < vout.size(); i++)
      vout[iinc(i, voind)]= (ovalue_type) 0;
    typename Matrix::block_cursor_type cursor= ma.bbegin(), end= ma.bend();
    for (; cursor != end; ++cursor) {
      size_t row= cursor.get_r(), col= cursor.get_c(), a= cursor.get_a(), b= cursor.get_b();
      for (size_t r= row; r < row+a; r++) {
	el_cursor_type el_cursor(cursor.el_cursor());
	ovalue_type tmp= vout[idec_wrt(r, mind, voind)]; // try to sum in register
	for (size_t c= col; c < col+b; c++)
	  tmp+= value(ma, *el_cursor++) * vin[idec_wrt(c, mind, viind)];
	vout[idec_wrt(r, mind, voind)]= tmp; } }
  }

} // namespace mtl

#endif // MTL_MAT_VEC_MULT_INCLUDE

