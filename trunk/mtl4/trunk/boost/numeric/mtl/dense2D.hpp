// $COPYRIGHT$

#ifndef MTL_DENSE2D_INCLUDE
#define MTL_DENSE2D_INCLUDE

#include "detail/base_cursor.hpp"
#include "detail/base_matrix.hpp"
#include "dim_type.hpp"

namespace mtl {

// cursor over every element
template <class ELT> //, class Offset>
class dense_el_cursor : public detail::base_matrix_cursor<const ELT*> {
public:
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type; // ?
  typedef detail::base_matrix_cursor<const ELT*> super;
  // typedef typename Offset::size_type    size_type;
  // typedef std::pair<size_type,size_type> pair_type;

  dense_cursor () {} 
  dense_cursor (pointer_type me, pointer_type b, dim_type d)
    : super(me, b, d) {}
 
protected:

};


  
  // M and N as template parameters might be considered later
template <class ELT, class Orientation>
class dense2D : public detail::base_matrix<ELT, Orientation> {
  typedef detail::base_matrix<ELT>      basem;
public:	
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type;
  typedef dense_el_cursor<ELT>          el_cursor_type;  
  typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;
  
  el_cursor_type ebegin() const {
    return el_cursor_type (data_const, data_const); }
  el_cursor_type eend() const {
    return el_cursor_type (data_const+nnz, data_const); }
  el_cursor_pair erange() const {
    return std::make_pair(ebegin(), eend()); }

protected:
  dim_type               dim;
  pointer_type           mbegin; // start address of matrix data
}; // dense2D


template <class ELT, class Orien>
std::size_t row( const dense2<ELT, Orien>&, const ELT* ) { 
  // ERROR !!! wrong type for Orien, replace with exception
  return 0; }

template <class ELT>
std::size_t row( const dense2D<ELT, row_major>& ma, const ELT* key ) { 
  return ma.offset(key) / ma.dim1(); }

template <class ELT>
std::size_t row( const dense2D<ELT, col_major>& ma, const ELT* key ) { 
  return ma.offset(key) % ma.dim2(); }


template <class ELT, class Orien>
std::size_t col( const dense2<ELT, Orien>&, const ELT* ) { 
  // ERROR !!! wrong type for Orien, replace with exception
  return 0; }

template <class ELT>
std::size_t col( const dense2D<ELT, row_major>& ma, const ELT* key ) { 
  return ma.offset(key) % ma.dim2(); }

template <class ELT>
std::size_t col( const dense2D<ELT, col_major>& ma, const ELT* key ) { 
  return ma.offset(key) / ma.dim1(); }


} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE
