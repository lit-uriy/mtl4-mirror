// $COPYRIGHT$

#ifndef MTL_PROPERTY_MAP_INCLUDE
#define MTL_PROPERTY_MAP_INCLUDE

namespace mtl {

  // will be removed probably and each matrix should provide its own method

  template <class Matrix, class Key>
  std::size_t row(const Matrix& ma, const Key& key) 
  {
    return ma.row(key); 
  }

  template <class Matrix, class Key>
  std::size_t col(const Matrix& ma, const Key& key) 
  {
    return ma.col(key); 
  }

  template <class Matrix, class Key>
  typename Matrix::value_type value(const Matrix& ma, const Key& key) 
  {
    return ma.value(key); 
  }

  template <class Matrix, class Key, class Val>
  void value(Matrix& ma, const Key& key, const Val& val) 
  {
    ma.value(key, val); 
  }



} // namespace mtl

#endif // MTL_PROPERTY_MAP_INCLUDE
