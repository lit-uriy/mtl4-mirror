// $COPYRIGHT$

#ifndef MTL_BASE_CURSOR_INCLUDE
#define MTL_BASE_CURSOR_INCLUDE



namespace mtl { namespace detail {

template <class Key>
class base_cursor {
public:
  typedef Key          key_type;
  typedef base_cursor  self;

  base_cursor () {} 
  base_cursor (key_type kk) : k(kk) {}

  key_type operator*() const {
    return k; }

  self& operator++ () { ++k; return *this; }
  self operator++ (int) { self tmp = *this; ++k; return tmp; }
  self& operator-- () { --k; return *this; }
  self operator-- (int) { self tmp = *this; --k; return tmp; }
  self& operator+=(int n) { k += n; return *this; }
  self& operator-=(int n) { k -= n; return *this; }
  
protected:
  key_type k;
}; // base_cursor

}} /* namespace mtl::detail */

#endif /* MTL_BASE_CURSOR_INCLUDE */
