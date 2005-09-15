// $COPYRIGHT$

#ifndef MTL_BASE_CURSOR_INCLUDE
#define MTL_BASE_CURSOR_INCLUDE

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


