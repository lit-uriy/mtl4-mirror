// $COPYRIGHT$

#ifndef MTL_BASE_MATRIX_INCLUDE
#define MTL_BASE_MATRIX_INCLUDE

#include <boost/numeric/mtl/dim_type.hpp>
#include <boost/numeric/mtl/base_types.hpp>

namespace mtl { namespace detail {
  using std::size_t;
  
  // base class for other matrices
  template <class ELT, class Orientation = mtl::row_major, class Dimension = mtl::dim_type>
  struct base_matrix 
  {
    typedef ELT                     value_type;
    typedef value_type*             pointer_type;
    typedef const value_type*       const_pointer_type;
    typedef pointer_type            key_type;
    typedef Dimension               dim_type;
    typedef Orientation             orientation;
  protected:
    value_type*                     data;      // pointer to matrix
    bool                            ext;       // whether pointer to external data or own
    dim_type                        dim;       // # of rows and columns
    size_t                          nnz;       // # of non-zeros, to be set by derived matrix
    orientation                     orien;     // objects are inherited, types not ;-)
    
    // allocate memory for contiguous formats
    // derived class is responsible that nnz is correctly set
    void allocate() 
    {
      if (! ext && data) delete data; 
      data = new value_type[nnz];
      ext = false;
    }
    
  public:
    base_matrix() : data(0), ext(false), nnz(0) {}

    // setting dimension, internal data but not yet allocated
    explicit base_matrix(mtl::dim_type d) : data(0), ext(false), dim(d), nnz(0) {}

    // setting dimension and reference to external data
    // nnz should be set by derived class 
    explicit base_matrix(mtl::dim_type d, value_type* a) : data(a), ext(true), dim(d), nnz(0) {}

    // same constructors for compile time matrix size
    // sets dimensions and pointer to external data
    explicit base_matrix(value_type* a) : data(a), ext(true), nnz(0) 
    { 
//       BOOST_ASSERT((dim_type::is_static));
    }

    // destruct if my own data (and allocated)
    ~base_matrix() 
    { 
      if (! ext && data) delete data; 
    }
    
    // number of rows
    size_t num_rows() const 
    {
      return dim.num_rows();
    }
    // numbef of colums
    size_t num_cols() const 
    {
      return dim.num_cols();
    }
  
  protected:
    // dispatched functions for major dimension
    size_t dim1(row_major) const 
    {
      return dim.num_rows();
    }
    size_t dim1(col_major) const 
    {
      return dim.num_cols();
    }
    size_t dim1(dia_major) const 
    {
      return dim.num_rows();
    }

    // dispatched functions for minor dimension
    size_t dim2(row_major) const 
    {
      return dim.num_cols();
    }
    size_t dim2(col_major) const 
    {
      return dim.num_rows();
    }
    size_t dim2(dia_major) const 
    {
      return dim.num_cols();
    } // or  2*cols-1 ???  
  
  public:
    // return major dimension
    size_t dim1() const 
    {
      return dim1(orien);
    }

    // return major dimension
    size_t dim2() const 
    {
      return dim2(orien);
    }

    // offset of key (pointer) w.r.t. data 
    // values must be stored consecutively
    size_t offset(const value_type* p) const 
    { 
      return p - data; 
    }

    // returns pointer to data
    pointer_type elements()
    {
      return data; 
    }

    // returns const pointer to data
    const_pointer_type elements() const 
    {
      return data; 
    }

    // returns copy of dim
    dim_type dim_ref() const 
    {
      return dim; 
    }

    // returns n-th value in consecutive memory
    // (whatever this means in the corr. matrix format)
    value_type val_n(size_t offset) const 
    { 
      return data[offset]; 
    }
    
  };

}} // namespace mtl::detail

#endif // MTL_BASE_MATRIX_INCLUDE
