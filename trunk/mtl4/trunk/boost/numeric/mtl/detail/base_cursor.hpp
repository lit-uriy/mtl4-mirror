// $COPYRIGHT$

#ifndef MTL_BASE_CURSOR_INCLUDE
#define MTL_BASE_CURSOR_INCLUDE

#include <boost/numeric/mtl/dim_type.hpp>

namespace mtl { namespace detail {

  // base class for different cursors, works with pointers and integers
  template <class Key>
  class base_cursor {
  public:
    typedef Key          key_type;
    typedef base_cursor  self;

    base_cursor () {} 
    base_cursor (key_type kk) : key(kk) {}

    key_type operator*() const 
    { 
      return key; 
    }

    self& operator++ () 
    { 
      ++key; return *this; 
    }
    self operator++ (int) 
    { 
      self tmp = *this; 
      ++key; 
      return tmp; 
    }
    self& operator-- () 
    { 
      --key; 
      return *this; 
    }
    self operator-- (int) 
    { 
      self tmp = *this; 
      --key; 
      return tmp; 
    }
    self& operator+=(int n) 
    { 
      key += n; 
      return *this; 
    }
    self& operator-=(int n) 
    { 
      key -= n; 
      return *this; 
    }
    bool operator==(const self& cc) const 
    {
      return key == cc.key; 
    }
    bool operator!=(const self& cc) const 
    {
      return !(*this == cc); 
    }
  
  protected:
    key_type key;
  }; // base_cursor



}} // namespace mtl::detail 

#endif // MTL_BASE_CURSOR_INCLUDE 



// = old code to throw away if new code works

// template <class Matrix>
// class base_matrix_cursor : base_cursor<typename Matrix::key_type> {
// public:
//   typedef typename Matrix::key_type     key_type;
//   typedef base_cursor<Key>              super;

//   base_matrix_cursor () : ma(0) {} 
//   base_matrix_cursor (key_type me_, const Matrix& ma_) :
//     super(me_), ma(&ma_) {}

//   // offset of key w.r.t. data
//   // std::size_t offset() const { return key-data; }
//   const Matrix*     ma;

// protected:
// //   key_type      data; // start address of matrix data
// //   dim_type      dim;
// }; // base_matrix_cursor
