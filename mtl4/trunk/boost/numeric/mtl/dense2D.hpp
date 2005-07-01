// $COPYRIGHT$

#ifndef MTL_DENSE2D_INCLUDE
#define MTL_DENSE2D_INCLUDE

#include <mtl/detail/base_cursor.hpp>
#include <mtl/detail/base_matrix.hpp>
#include <mtl/dim_type.hpp>
#include <mtl/base_types.hpp>

namespace mtl {

// cursor over every element
template <class ELT> //, class Offset>
class dense_el_cursor : public detail::base_cursor<const ELT*> {
public:
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type; // ?
  typedef detail::base_cursor<const ELT*> super;

  dense_el_cursor () {} 
  dense_el_cursor (pointer_type me) : super(me) {}
//   dense_cursor (pointer_type me, pointer_type b, dim_type d)
//     : super(me, b, d) {}
};


  
  // M and N as template parameters might be considered later
template <class ELT, class Orientation>
class dense2D : public detail::base_matrix<ELT, Orientation> {
  typedef detail::base_matrix<ELT, Orientation>      super;
  typedef dense2D                       self;
public:	
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type;
  typedef dense_el_cursor<ELT>          el_cursor_type;  
  typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;
  
  dense2D() : super() {}
  dense2D(dim_type d) : super(d) {}
  dense2D(dim_type d, value_type* a) : super(d, a) {
    nnz = d.rows() *d.cols(); }

  template <class InputIterator>
  dense2D(dim_type d, InputIterator first, InputIterator last) : super(d) {
    nnz = d.rows() *d.cols();
    data = new value_type[nnz];
    value_type* p = data;
    for (std::size_t i = 0; i < nnz; i++) *p++ = *first++; 
    // check if first == last otherwise throw exception
  }

  el_cursor_type ebegin() const {
    return el_cursor_type (data_ref()); }
  el_cursor_type eend() const {
    return el_cursor_type (data_ref()+nnz); }
  el_cursor_pair erange() const {
    return std::make_pair(ebegin(), eend()); }

protected:
}; // dense2D

  // row and col will be computed by some indexer object later
  // fortran enumeration is not taken into account yet
template <class ELT, class Orien>
std::size_t row( const dense2D<ELT, Orien>&, const ELT* ) { 
  // ERROR !!! wrong type for Orien, replace with exception
  return 0; }

template <class ELT>
std::size_t row( const dense2D<ELT, row_major>& ma, const ELT* key ) { 
  return ma.offset(key) / ma.dim2(); }

template <class ELT>
std::size_t row( const dense2D<ELT, col_major>& ma, const ELT* key ) { 
  return ma.offset(key) % ma.dim2(); }


template <class ELT, class Orien>
std::size_t col( const dense2D<ELT, Orien>&, const ELT* ) { 
  // ERROR !!! wrong type for Orien, replace with exception
  return 0; }

template <class ELT>
std::size_t col( const dense2D<ELT, row_major>& ma, const ELT* key ) { 
  return ma.offset(key) % ma.dim2(); }

template <class ELT>
std::size_t col( const dense2D<ELT, col_major>& ma, const ELT* key ) { 
  return ma.offset(key) / ma.dim2(); }

template <class ELT, class Orien>
ELT value( const dense2D<ELT, Orien>&, const ELT* key ) {
  return *key; }

} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE
