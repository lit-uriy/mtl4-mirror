// $COPYRIGHT$

#ifndef MTL_PROPERTY_MAP_INCLUDE
#define MTL_PROPERTY_MAP_INCLUDE

namespace mtl {

template <class Matrix, class Key>
std::size_t row(const Matrix& ma, const Key& key) {
  return ma.row(key); }

template <class Matrix, class Key>
std::size_t column(const Matrix& ma, const Key& key) {
  return ma.column(key); }

template <class Matrix, class Key>
typename Matrix::value_type value(const Matrix& ma, const Key& key) {
  return ma.value(key); }

template <class Matrix, class Key, class Val>
void value(const Matrix& ma, const Key& key, const Value& val) {
  return ma.value(key, val); }



} // namespace mtl

#endif // MTL_PROPERTY_MAP_INCLUDE
