// $COPYRIGHT$

#ifndef MTL_DENSE2D_INCLUDE
#define MTL_DENSE2D_INCLUDE

#include <mtl/detail/base_cursor.hpp>
#include <mtl/detail/base_matrix.hpp>
#include <mtl/dim_type.hpp>
#include <mtl/base_types.hpp>
#include <mtl/intrinsics.hpp>

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


template <class Matrix>
class dense2D_indexer {
  typedef Matrix                        matrix_type;
  typedef typename Matrix::orientation  orientation;
  typedef typename Matrix::ind          ind;
  typedef typename Matrix::key_type     key_type;

  std::size_t _offset(const matrix_type& ma, std::size_t r, std::size_t c, row_major) const {
    return r * ma.dim2() + c; }
  std::size_t _offset(const matrix_type& ma, std::size_t r, std::size_t c, col_major) const {
    return c * ma.dim2() + r; }

  std::size_t _row(const matrix_type& ma, const key_type& key, row_major) const {
    return ma.offset(key) / ma.dim2(); }
  std::size_t _row(const matrix_type& ma, const key_type& key, col_major) const {
    return ma.offset(key) % ma.dim2(); }

  std::size_t _col(const matrix_type& ma, const key_type& key, row_major) const {
    return ma.offset(key) % ma.dim2(); }
  std::size_t _col(const matrix_type& ma, const key_type& key, col_major) const {
    return ma.offset(key) / ma.dim2(); }

 public:
  // dealing with fortran indices here (to do it only once) and orientation above
  std::size_t operator() (const matrix_type& ma, std::size_t r, std::size_t c) const {
    return _offset(ma, idec(r, ind()), idec(c, ind()), orientation()); }

  std::size_t row(const matrix_type& ma, const key_type& key) const {
    return iinc( _row(ma, key, orientation()), ind() ); }

  std::size_t col(const matrix_type& ma, const key_type& key) const {
    return iinc( _col(ma, key, orientation()), ind() ); }
};


  
  // M and N as template parameters might be considered later
template <class ELT, class Orientation= row_major, class Indexing= c_index>
class dense2D : public detail::base_matrix<ELT, Orientation> {
  typedef detail::base_matrix<ELT, Orientation>      super;
  typedef dense2D                       self;
public:	
  typedef Orientation                   orientation;
  typedef Indexing                      ind;
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type;
  typedef pointer_type                  key_type;
  typedef dense_el_cursor<ELT>          el_cursor_type;  
  typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;
  typedef dense2D_indexer<self>         indexer_type;
  
  dense2D() : super() {}
  dense2D(dim_type d) : super(d) {} // , my_indexer(*this) {}
  dense2D(dim_type d, value_type* a) : super(d, a) { nnz= d.rows() *d.cols();}
  dense2D(dim_type d, value_type value) : super(d) {
    nnz= d.rows() *d.cols();
    data = new value_type[nnz];
    value_type* p = data;
    for (std::size_t i = 0; i < nnz; i++) *p++ = value; 
  }

  template <class InputIterator>
  dense2D(dim_type d, InputIterator first, InputIterator last) : super(d) {
    nnz= d.rows() *d.cols();
    data = new value_type[nnz];
    value_type* p = data;
    for (std::size_t i = 0; i < nnz; i++) *p++ = *first++; 
    // check if first == last otherwise throw exception
  }

  friend class indexer_type;

  el_cursor_type ebegin() const {
    return el_cursor_type (data_ref()); }
  el_cursor_type eend() const {
    return el_cursor_type (data_ref()+nnz); }
  el_cursor_pair erange() const {
    return std::make_pair(ebegin(), eend()); }

  value_type operator() (std::size_t r, std::size_t c) const {
    return data[indexer(*this, r, c)]; }

  std::size_t row(const key_type& key) const {
    return indexer.row(*this, key); }
  std::size_t col(const key_type& key) const {
    return indexer.col(*this, key); }
  value_type value(const key_type& key) const {
    return *key; }
  void value(const key_type& key, const value_type& value) {
    * const_cast<value_type *>(key)= value; }

protected:
  const indexer_type  indexer;
}; // dense2D

// declare as fortran indexed if so
// template <class ELT, class Orientation>
// struct is_fortran_indexed<dense2D<ELT, Orientation, f_index> > {
//   static const bool value= true; };
// should be done automatically

template <class ELT, class Orientation, class Index>
struct is_mtl_type<dense2D<ELT, Orientation, Index> > {
  static const bool value= true; };

} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE
