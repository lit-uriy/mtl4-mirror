// $COPYRIGHT$

#ifndef MTL_TRANSPOSED_VIEW_INCLUDE
#define MTL_TRANSPOSED_VIEW_INCLUDE

#include <mtl/base_types.hpp>

namespace mtl {

template <class Matrix>
class transposed_view {
  typedef transposed_view               self;
  typedef Matrix                        other;
public:	
  typedef typename transposed_orientation<typename Matrix::orientation>::type orientation;
  typedef typename Matrix::ind            ind;
  typedef typename Matrix::value_type     value_type;
  typedef typename Matrix::pointer_type   pointer_type;
  typedef typename Matrix::key_type       key_type;
  typedef typename Matrix::el_cursor_type el_cursor_type;
  typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;

  transposed_view (other& ref_) : ref(ref_) {}

  el_cursor_type ebegin() const { return ref.ebegin(); }
  el_cursor_type eend() const { return ref.eend(); }
  el_cursor_pair erange() const { return std::make_pair(ebegin(), eend()); }
  
  value_type operator() (std::size_t r, std::size_t c) { return ref(c, r); }

  std::size_t dim1() const { return ref.dim2(); }
  std::size_t dim2() const { return ref.dim1(); }
  std::size_t row(const key_type& key) const { return ref.col(key); }
  std::size_t col(const key_type& key) const { return ref.row(key); }
  value_type value(const key_type& key) const { return ref.value(key); }
  // no overwriting of data at the moment
  void value(const key_type& key, const value_type& value) { ref.value(key, value); }

  std::size_t offset(const value_type* p) const { return ref.offset(p); }
  pointer_type data_ref() const {return ref.data_ref(); }
  value_type val_n(std::size_t offset) const { return ref.val_n(offset); }

  dim_type dim_ref() const {return ref.dim_ref().transpose(); }

protected:
  other& ref;
};

} // namespace mtl

#endif // MTL_TRANSPOSED_VIEW_INCLUDE
