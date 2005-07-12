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
  dense_el_cursor (pointer_type me_) : super(me_) {}
};

// // cursor over every element
// template <class Matrix> //, class Offset>
// class dense_el_cursor : 
//     public detail::base_matrix_cursor<Matrix> {
// public:
//   typedef detail::base_matrix_cursor<Matrix>   super;
//   typedef super::value_type                    value_type;
//   typedef super::key_type                      key_type;

//   dense_el_cursor () {} 
//   dense_el_cursor (key_type me_, const Matrix& ma_) : super(me_, ma) {}
// };



template <class Matrix>
class dense2D_indexer {
  typedef Matrix                        matrix_type;
  typedef typename Matrix::orientation  orientation;
  typedef typename Matrix::key_type     key_type;

  std::size_t row_(const matrix_type& ma, const key_type& key, row_major) const {
    return ma.offset(key) / ma.dim2(); }
  std::size_t row_(const matrix_type& ma, const key_type& key, col_major) const {
    return ma.offset(key) % ma.dim2(); }
  std::size_t col_(const matrix_type& ma, const key_type& key, row_major) const {
    return ma.offset(key) % ma.dim2(); }
  std::size_t col_(const matrix_type& ma, const key_type& key, col_major) const {
    return ma.offset(key) / ma.dim2(); }

 public:
  std::size_t row(const matrix_type& ma, const key_type& key) const {
    return _row(ma, key, orientation()); }
  std::size_t column(const matrix_type& ma, const key_type& key) const {
    return _column(ma, key, orientation()); }

 };


  
  // M and N as template parameters might be considered later
template <class ELT, class Orientation, class Indexing>
class dense2D : public detail::base_matrix<ELT, Orientation> {
  typedef detail::base_matrix<ELT, Orientation>      super;
  typedef dense2D                       self;
public:	
  typedef Orientation                   orientation;
  typedef Indexing                      indexing;
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type;
  typedef pointer_type                  key_type;
  typedef dense_el_cursor<ELT>          el_cursor_type;  
  typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;
  typedef dense2D_indexer<self>         indexer;
  
  dense2D() : super() {}
  dense2D(dim_type d) : super(d) {} // , my_indexer(*this) {}
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

  friend indexer;

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

template <class ELT, class Orien>
void value( const dense2D<ELT, Orien>&, const ELT* key, const ELT& val ) {
  * const_cast<ELT*>(key) = val; }

} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE
