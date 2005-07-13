// $COPYRIGHT$

#ifndef MTL_BASE_MATRIX_INCLUDE
#define MTL_BASE_MATRIX_INCLUDE

#include <mtl/dim_type.hpp>
#include <mtl/base_types.hpp>

namespace mtl { namespace detail {

template <class ELT, class Orientation>
class base_matrix {
public:
  typedef ELT                     value_type;
  typedef const value_type*       pointer_type;
  typedef pointer_type            key_type;
  typedef Orientation             orientation;

  base_matrix() : data(0), ext(false), nnz(0) {}
  base_matrix(dim_type d) : data(0), ext(false), dim(d), nnz(0) {}
  base_matrix(dim_type d, value_type* a) : data(a), ext(true), dim(d), nnz(0) {}
  ~base_matrix() { delete data; }

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
  pointer_type data_ref() const {return data; }
  dim_type dim_ref() const {return dim; }
  value_type val_n(std::size_t offset) const { return data[offset]; }

protected:
  value_type*                     data;   // pointer to matrix
  // static const value_type* const& data_const(data);  // to not pass mutable outside
  bool                            ext;
  dim_type                        dim;       // # of rows and columns
  std::size_t                     nnz;       // # of non-zeros, to be set by derived matrix
  orientation                     orien;     // objects are inherited, types not ;-)
};

}} // namespace mtl::detail

#endif // MTL_BASE_MATRIX_INCLUDE
