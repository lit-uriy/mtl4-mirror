// $COPYRIGHT$

#ifndef MTL_FRACTALU_INCLUDE
#define MTL_FRACTALU_INCLUDE

#include <mtl/detail/base_cursor.hpp>
#include <mtl/detail/base_matrix.hpp>
#include <mtl/dim_type.hpp>
#include <mtl/base_types.hpp>
#include <mtl/intrinsics.hpp>

namespace mtl {

// cursor over every element
template <class ELT> //, class Offset>
class fractalu_el_cursor : public detail::base_cursor<const ELT*> {
public:
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type; // ?
  typedef detail::base_cursor<const ELT*> super;

  fractalu_el_cursor () {} 
  fractalu_el_cursor (pointer_type me_) : super(me_) {}
};

template <class Matrix>
class fractalu_indexer {
  typedef Matrix                        matrix_type;
  typedef typename Matrix::key_type     key_type;
 public: 
  std::size_t operator() (const matrix_type& ma, std::size_t r, std::size_t c) const {
    std::size_t a= ma.rows(), b= ma.cols(), n= 0, af, as, bf, bs, nf= ma.nf;
    while (a > nf && b > nf) {
      af= a/2; as= a-af; bf= b/2; bs= b-bf;
      if (r < af)
	if (c < bf) a= af, b= bf;
	else r-= af, n+= af*bf, a= as, b= bf;
      else {
	n+= af*b;
	if (c >= bf) r-= af, c-= bf, a= as, b= bs;
	else c-= bf, n+= as*bs, a= af, b= bs;
      } }
    return n + r*b + c;
  }

  std::pair<std::size_t, std::size_t> row_col (const matrix_type& ma, std::size_t n) const {
    std::size_t a= ma.rows(), b= ma.cols(), r= 0, c= 0, af, as, bf, bs, nf= ma.nf;
    while (a > nf && b > nf) {
      af= a/2; as= a-af; bf= b/2; bs= b-bf;
      if (n < af*b)
	if (n < af*bf) a= af, b= bf;
	else r+= af, n-= af*bf, a= as, b= bf;
      else {
	n-= af*b;
	if (n < as*bs) r+= af, c+= bf, a= as, b= bs;
	else c+= bf, n-= as*bs, a= af, b= bs;
      } }
    return std::make_pair(r + n/b, c + n%b);
  }

  std::size_t row(const matrix_type& ma, std::size_t n) const {
    return row_col(ma, n).first; }
  std::size_t row(const matrix_type& ma, const key_type& key) const {
    return row(ma, ma.offset(key)); }
  
  std::size_t col(const matrix_type& ma, std::size_t n) const {
    return row_col(ma, n).second; }
  std::size_t col(const matrix_type& ma, const key_type& key) const {
    return col(ma, ma.offset(key)); }
}; // fractalu_cursor

template <class ELT, std::size_t NF= 8>
class fractalu : public detail::base_matrix<ELT, row_major> {
  typedef detail::base_matrix<ELT, row_major>      super;
  typedef fractalu                      self;
public:	
  typedef c_index                       ind;
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type;
  typedef pointer_type                  key_type;
  typedef fractalu_el_cursor<ELT>       el_cursor_type;  
  typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;
  typedef fractalu_indexer<self>        indexer_type;
  static const std::size_t              nf= NF;
  
  // does not work yet if row < 2^k && col > 2^k and vice versa 
  fractalu() : super() {}
  fractalu(dim_type d) : super(d) {} // , my_indexer(*this) {}
  fractalu(dim_type d, value_type* a) : super(d, a) { nnz= d.rows() *d.cols();}
  fractalu(dim_type d, value_type value) : super(d) {
    nnz= d.rows() *d.cols();
    data = new value_type[nnz];
    value_type* p = data;
    for (std::size_t i = 0; i < nnz; i++) *p++ = value; 
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
}; // fractalu

} // namespace mtl

#endif // MTL_FRACTALU_INCLUDE
