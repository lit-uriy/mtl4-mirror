// $COPYRIGHT$

#ifndef MTL_BASE_CURSOR_INCLUDE
#define MTL_BASE_CURSOR_INCLUDE

#include <mtl/dim_type.hpp>

namespace mtl { namespace detail {

template <class Key>
class base_cursor {
public:
  typedef Key          key_type;
  typedef base_cursor  self;

  base_cursor () {} 
  base_cursor (key_type kk) : key(kk) {}

  key_type operator*() const { return key; }

  self& operator++ () { ++key; return *this; }
  self operator++ (int) { self tmp = *this; ++key; return tmp; }
  self& operator-- () { --key; return *this; }
  self operator-- (int) { self tmp = *this; --key; return tmp; }
  self& operator+=(int n) { key += n; return *this; }
  self& operator-=(int n) { key -= n; return *this; }
  bool operator==(const self& cc) const {return key == cc.key; }
  bool operator!=(const self& cc) const {return !(*this == cc); }
  
protected:
  key_type key;
}; // base_cursor



// template <class Key>
// class base_matrix_cursor : base_cursor<Key> {
// public:
//   typedef Key                  key_type;
//   typedef base_cursor<Key>     super;

//   base_matrix_cursor () {} 
//   base_matrix_cursor (key_type me, key_type mb, dim_type d) :
//     super(me), data(mb), dim(d) {}

//   // offset of key w.r.t. data
//   std::size_t offset() const { return key-data; }
// protected:
//   key_type      data; // start address of matrix data
//   dim_type      dim;
// }; // base_matrix_cursor

}} // namespace mtl::detail 

#endif // MTL_BASE_CURSOR_INCLUDE 
