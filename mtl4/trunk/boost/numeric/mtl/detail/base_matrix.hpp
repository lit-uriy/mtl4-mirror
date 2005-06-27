// $COPYRIGHT$

#ifndef MTL_BASE_MATRIX_INCLUDE
#define MTL_BASE_MATRIX_INCLUDE

#include "dim_type.hpp"

namespace mtl { namespace detail {

template <class ELT, class Orientation>
class base_matrix {
public:
  typedef ELT                     value_type;
  typedef Orientation             orientation;

  base_matrix() : data(0), ext(false) {}
  base_matrix(dim_type d) : data(0), ext(false), dim(d) {}
  base_matrix(dim_type d, value_type* a) : data(a), ext(true), dim(d) {}

  std::size_t rows() const {return dim.rows();}
  std::size_t columns() const {return dim.columns();}
  
protected:
  std::size_t dim1(row_major) const {return dim.rows();}
  std::size_t dim1(col_major) const {return dim.cols();}
  std::size_t dim1(dia_major) const {return dim.rows();}

  std::size_t dim2(row_major) const {return dim.cols();}
  std::size_t dim2(col_major) const {return dim.rows();}
  std::size_t dim2(dia_major) const {return dim.cols();} // or  2*cols-1 ???
  
  
public:
  std::size_t dim1() const {return dim1(orien);}
  std::size_t dim2() const {return dim2(orien);}
  // offset of key (pointer) w.r.t. data
  std::size_t offset(const value_type* p) const { return p-data; }

protected:
  value_type*                     data;   // pointer to matrix
  const value_type* const&        data_const(data);  // to not pass mutable
  bool                            ext;
  dim_type                        dim;       // # of rows and columns
  std::size_t                     nnz;       // # of non-zeros (size)
  orientation                     orien;
};

}} // namespace mtl::detail

#endif // MTL_BASE_MATRIX_INCLUDE
