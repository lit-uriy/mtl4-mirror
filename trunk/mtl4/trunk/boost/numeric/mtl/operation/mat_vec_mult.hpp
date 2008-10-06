// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MAT_VEC_MULT_INCLUDE
#define MTL_MAT_VEC_MULT_INCLUDE

#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/detail/index.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>



#include <iostream>

namespace mtl {

// Dense matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::dense)
{
    // Naive implementation, will be moved to a functor and complemented with more efficient ones

    using math::zero;
    if (size(w) == 0) return;

    if (Assign::init_to_zero) set_to_zero(w);

    typedef typename Collection<VectorOut>::value_type value_type;

    for (unsigned i= 0; i < num_rows(a); i++) {
	value_type tmp= zero(w[i]);
	for (unsigned j= 0; j < num_cols(a); j++) 
	    tmp+= a[i][j] * v[j];
	Assign::update(w[i], tmp);
    }
}



// Sparse matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::sparse)
{
    smat_cvec_mult(a, v, w, Assign(), typename OrientedCollection<Matrix>::orientation());
}



// Sparse row-major matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void smat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::row_major)
{
    using namespace tag; 
    using traits::range_generator;  
    using math::zero;

    typedef typename range_generator<row, Matrix>::type       a_cur_type;    
    typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
    typename traits::col<Matrix>::type                        col_a(a); 
    typename traits::const_value<Matrix>::type                value_a(a); 

    if (Assign::init_to_zero) set_to_zero(w);

    typedef typename Collection<VectorOut>::value_type        value_type;

    a_cur_type ac= begin<row>(a), aend= end<row>(a);
    for (unsigned i= 0; ac != aend; ++ac, ++i) {
	value_type tmp= zero(w[i]);
	for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
	    tmp+= value_a(*aic) * v[col_a(*aic)];	
	Assign::update(w[i], tmp);
    }
}

// Sparse column-major matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void smat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::col_major)
{
	using namespace tag;
	using traits::range_generator;  
        typedef typename range_generator<col, Matrix>::type       a_cur_type;             
        typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            

	typename traits::row<Matrix>::type                        row_a(a); 
	typename traits::const_value<Matrix>::type                value_a(a); 

	if (Assign::init_to_zero) set_to_zero(w);

	unsigned rv= 0; // traverse all rows of v
	for (a_cur_type ac= begin<col>(a), aend= end<col>(a); ac != aend; ++ac, ++rv) {
	    typename Collection<VectorIn>::value_type    vv= v[rv]; 
	    for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
		Assign::update(w[row_a(*aic)], value_a(*aic) * vv);
	}
}


} // namespace mtl



#if 0


// obselete code -> to be deleted soon

namespace mtl {
  using std::size_t;

  // general matrix vector product for dense matrices (might be slow)
  // for dense2D: row major ~ 4 times faster, column major ~ 3 times slower
  template <class Matrix, class Vector_in, class Vector_out> 
  void dense_mat_vec_mult(const Matrix& ma, const Vector_in& vin, Vector_out& vout) 
  {
    // if (ma.num_rows() != vout.size()) throw something;
    // if (ma.num_cols() != vin.size()) throw something else;

    typename index::which_index<Matrix>::type m    atrix_index;
    typename index::which_index<Vector_in>::type   vin_index;
    typename index::which_index<Vector_out>::type  vout_index;



    size_t mi= 0, mrows= ma.num_rows(), mj_start= 0, mcols= ma.num_cols(), vi= 0, vj_start= 0;
    

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
    // if (ma.num_rows() != vout.size()) throw something;
    // if (ma.num_cols() != vin.size()) throw something else;
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
    // if (ma.num_rows() != vout.size()) throw something;
    // if (ma.num_cols() != vin.size()) throw something else;
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

#endif

#endif // MTL_MAT_VEC_MULT_INCLUDE

