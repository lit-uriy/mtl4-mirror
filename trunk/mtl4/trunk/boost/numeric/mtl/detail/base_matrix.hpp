// $COPYRIGHT$

#ifndef MTL_BASE_MATRIX_INCLUDE
#define MTL_BASE_MATRIX_INCLUDE

#include "dimension.hpp"

namespace mtl { namespace detail {

template <class ELT, class Orientation>
class base_matrix {
public:
  typedef ELT                     value_type;
  typedef Orientation             orientation;

  base_matrix() : my_data(0), external(false) {}
  base_matrix(dim_type d) : my_data(0), external(false), dim(d) {}
  base_matrix(dim_type d, value_type* a) : my_data(a), external(true), dim(d) {}
  
protected:
  value_type*                     my_data;   // pointer to matrix
  const value_type* const&        my_data_const(my_data);  // to not pass mutable
  dim_type                        dim;       // # of rows and columns
  int                             nnz;       // # of non-zeros (size)
};

}} // namespace mtl::detail

#endif // MTL_BASE_MATRIX_INCLUDE
