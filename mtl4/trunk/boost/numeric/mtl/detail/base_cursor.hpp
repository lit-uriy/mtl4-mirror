// $COPYRIGHT$

#ifndef MTL_BASE_CURSOR_INCLUDE
#define MTL_BASE_CURSOR_INCLUDE

#include "dim_type.hpp"

namespace mtl { namespace detail {

template <class Key>
class base_cursor {
public:
  typedef Key          key_type;
  typedef base_cursor  self;

  base_cursor () {} 
  base_cursor (key_type kk) : key(kk) {}

  key_type operator*() const { return k; }

  self& operator++ () { ++k; return *this; }
  self operator++ (int) { self tmp = *this; ++k; return tmp; }
  self& operator-- () { --k; return *this; }
  self operator-- (int) { self tmp = *this; --k; return tmp; }
  self& operator+=(int n) { k += n; return *this; }
  self& operator-=(int n) { k -= n; return *this; }
  
protected:
  key_type key;
}; // base_cursor



template <class Key>
class base_matrix_cursor : base_cursor<Key> {
public:
  typedef Key                  key_type;
  typedef base_cursor<Key>     super;

  base_matrix_cursor () {} 
  base_matrix_cursor (key_type me, key_type mb, dim_type d) :
    super(me), data(mb), dim(d) {}

  // offset of key w.r.t. data
  std::size_t offset() const { return key-data; }
protected:
  key_type      data; // start address of matrix data
  dim_type      dim;
}; // base_matrix_cursor

}} // namespace mtl::detail 

#endif // MTL_BASE_CURSOR_INCLUDE 
