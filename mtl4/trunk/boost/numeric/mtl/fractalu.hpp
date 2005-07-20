// $COPYRIGHT$

#ifndef MTL_FRACTALU_INCLUDE
#define MTL_FRACTALU_INCLUDE

#include <mtl/detail/base_cursor.hpp>
#include <mtl/detail/base_matrix.hpp>
#include <mtl/dim_type.hpp>
#include <mtl/base_types.hpp>
#include <mtl/intrinsics.hpp>
#include <boost/tuple/tuple.hpp>

namespace mtl {

  using std::size_t;

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

// cursor over blocks
  template <class ELT, class Matrix> 
class fractalu_block_cursor {
  typedef fractalu_block_cursor         self;
  size_t                                n, r, c, a, b;
  typedef Matrix                        matrix_type;
  const matrix_type*                    ma;
public:
  typedef ELT                           value_type;
  fractalu_block_cursor() : n(0), r(0), c(0), a(0), b(0), ma(0) {}
  fractalu_block_cursor(size_t n_, const matrix_type& ma_) : n(n_), ma(&ma_) {
    size_t ntest;
    boost::tie(r, c, ntest, a, b)= ma->indexer.row_col_block(*ma, n); 
    // if ntest > 0 not at the beginning of a block -> exception
  }

  self& operator* () { return *this; }
  bool operator==(const self& cc) const {return n == cc.n; }
  bool operator!=(const self& cc) const {return !(*this == cc); }

  self& operator++ () {
    size_t ntest; n+= a*b;
    boost::tie(r, c, ntest, a, b)= ma->indexer.row_col_block(*ma, n); return *this; 
    // if ntest > 0 not at the beginning of a block -> exception
  }
  self operator++ (int) { self tmp= *this; operator++(); return tmp; }
    
  self& operator-- () {
    size_t ncorr; n--; // n at the end of previous block
    boost::tie(r, c, ncorr, a, b)= ma->indexer.row_col_block(*ma, n); 
    n-= ncorr;  return *this; // go to the beginning of the block
  }
  self operator-- (int) { self tmp= *this; operator--(); return tmp; }

  size_t get_n() const { return n; }
  size_t get_r() const { return r; }
  size_t get_c() const { return c; }
  size_t get_a() const { return a; }
  size_t get_b() const { return b; }

  value_type value() const { return ma->val_n(n); }
};

  // returning the row, column and value of a block's first entry (for completeness)
  template <class ELT, class Matrix> 
  size_t row (const Matrix&, const fractalu_block_cursor<ELT, Matrix>& cu) { 
    return cu.get_r(); }
  template <class ELT, class Matrix> 
  size_t col (const Matrix&, const fractalu_block_cursor<ELT, Matrix>& cu) { 
    return cu.get_c(); }
  template <class ELT, class Matrix> 
  ELT value (const Matrix&, const fractalu_block_cursor<ELT, Matrix>& cu) { 
    return cu.value(); }

    

template <class Matrix>
class fractalu_indexer {
  typedef Matrix                        matrix_type;
  typedef typename Matrix::key_type     key_type;
 public: 
  size_t operator() (const matrix_type& ma, size_t r, size_t c) const;

  boost::tuple<size_t, size_t, size_t, size_t, size_t> row_col_block (const matrix_type& ma, size_t n) const;

  std::pair<size_t, size_t> row_col (const matrix_type& ma, size_t n) const {
    size_t r, c, nrest, a, b;
    boost::tie(r, c, nrest, a, b)= row_col_block(ma, n);
    return std::make_pair(r + nrest/b, c + nrest%b); }

  size_t row(const matrix_type& ma, size_t n) const {
    return row_col(ma, n).first; }
  size_t row(const matrix_type& ma, const key_type& key) const {
    return row(ma, ma.offset(key)); }
  
  size_t col(const matrix_type& ma, size_t n) const {
    return row_col(ma, n).second; }
  size_t col(const matrix_type& ma, const key_type& key) const {
    return col(ma, ma.offset(key)); }
}; // fractalu_cursor



  template <class Matrix>
  inline size_t fractalu_indexer<Matrix>::operator() (const matrix_type& ma, size_t r, size_t c) const {
    size_t a= ma.rows(), b= ma.cols(), n= 0, af, as, bf, bs, nf= ma.nf;
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

  template <class Matrix>
  inline boost::tuple<size_t, size_t, size_t, size_t, size_t>
  fractalu_indexer<Matrix>::row_col_block (const matrix_type& ma, size_t n) const {
    size_t a= ma.rows(), b= ma.cols(), r= 0, c= 0, af, as, bf, bs, nf= ma.nf;
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
    return boost::make_tuple(r, c, n, a, b);
  }


template <class ELT, size_t NF= 8>
class fractalu : public detail::base_matrix<ELT, row_major> {
  typedef detail::base_matrix<ELT, row_major>      super;
  typedef fractalu                      self;
public:	
  typedef c_index                       ind;
  typedef ELT                           value_type;
  typedef const value_type*             pointer_type;
  typedef pointer_type                  key_type;
  typedef fractalu_el_cursor<ELT>       el_cursor_type;  
  typedef std::pair<el_cursor_type, el_cursor_type>  el_cursor_pair;
  typedef fractalu_block_cursor<ELT, self>           block_cursor_type;  
  typedef std::pair<block_cursor_type, block_cursor_type>  block_cursor_pair;
  typedef fractalu_indexer<self>        indexer_type;
  static const size_t                   nf= NF;
  
  // does not work yet if row < 2^k && col > 2^k and vice versa 
  fractalu() : super() {}
  fractalu(dim_type d) : super(d) {} // , my_indexer(*this) {}
  fractalu(dim_type d, value_type* a) : super(d, a) { nnz= d.rows() *d.cols();}
  fractalu(dim_type d, value_type value) : super(d) {
    nnz= d.rows() *d.cols();
    data = new value_type[nnz];
    value_type* p = data;
    for (size_t i = 0; i < nnz; i++) *p++ = value; 
  }

  friend class indexer_type;
  friend class block_cursor_type;

  value_type operator() (size_t r, size_t c) const {
    return data[indexer(*this, r, c)]; }

  el_cursor_type ebegin() const {
    return el_cursor_type (data_ref()); }
  el_cursor_type eend() const {
    return el_cursor_type (data_ref()+nnz); }
  el_cursor_pair erange() const {
    return std::make_pair(ebegin(), eend()); }

  block_cursor_type bbegin() const {
    return block_cursor_type (0, *this); }
  block_cursor_type bend() const {
    return block_cursor_type (nnz, *this); }
  block_cursor_pair brange() const {
    return std::make_pair(bbegin(), bend()); }

  size_t row(const key_type& key) const {
    return indexer.row(*this, key); }
  size_t col(const key_type& key) const {
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
