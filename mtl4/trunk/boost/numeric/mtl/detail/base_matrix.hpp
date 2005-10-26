// $COPYRIGHT$

#ifndef MTL_BASE_MATRIX_INCLUDE
#define MTL_BASE_MATRIX_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/index.hpp>

namespace mtl { namespace detail {
using std::size_t;
  
// Base class for other matrices
// will certainly be splitted multiple classes later when more matrices will be supported
template <class Elt, class Parameters>
struct base_matrix 
{
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    typedef Elt                     value_type;
    typedef value_type*             pointer_type;
    typedef const value_type*       const_pointer_type;
    typedef pointer_type            key_type;
  protected:
    value_type*                     data;      // pointer to matrix
    bool                            ext;       // whether pointer to external data or own
    dim_type                        dim;       // # of rows and columns
    size_t                          nnz;       // # of non-zeros, to be set by derived matrix
    
    // allocate memory for contiguous formats
    // derived class is responsible that nnz is correctly set
    void allocate() 
    {
      if (! ext && data) delete[] data; 
      data = new value_type[nnz];
      ext = false;
    }
    
  public:
    base_matrix() : data(0), ext(false), nnz(0) {}

    // setting dimension, internal data but not yet allocated
    explicit base_matrix(mtl::non_fixed::dimensions d) : data(0), ext(false), dim(d), nnz(0) {}

    // setting dimension and reference to external data
    // nnz should be set by derived class 
    explicit base_matrix(mtl::non_fixed::dimensions d, value_type* a) 
      : data(a), ext(true), dim(d), nnz(0) {}

    // same constructors for compile time matrix size
    // sets dimensions and pointer to external data
    explicit base_matrix(value_type* a) : data(a), ext(true), nnz(0) 
    { 
      BOOST_STATIC_ASSERT((dim_type::is_static));
    }

    // destruct if my own data (and allocated)
    ~base_matrix() 
    { 
      if (! ext && data) delete[] data; 
    }
    
    // number of rows
    size_t num_rows() const 
    {
      return dim.num_rows();
    }
    // First row taking indexing into account
    size_t begin_row() const 
    {
      return index::change_to(index_type(), 0);
    }
    // Past-end row taking indexing into account
    size_t end_row() const 
    {
      return index::change_to(index_type(), num_rows());
    }

    // number of colums
    size_t num_cols() const 
    {
      return dim.num_cols();
    }
    // First column taking indexing into account
    size_t begin_col() const 
    {
      return index::change_to(index_type(), 0);
    }
    // Past-end column taking indexing into account
    size_t end_col() const 
    {
      return index::change_to(index_type(), num_cols());
    }


    // number of elements
    size_t num_elements() const
    {
      return nnz;
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

    // dispatched functions for minor dimension
    size_t dim2(row_major) const 
    {
      return dim.num_cols();
    }
    size_t dim2(col_major) const 
    {
      return dim.num_rows();
    }
  
    // Dispatched functions for major
    // Trailing _ due to conflicts with macro major
    size_t major_(size_t r, size_t, row_major) const
    {
	return r; 
    }
    size_t major_(size_t, size_t c, col_major) const
    {
	return c; 
    }    

  public:
    // return major dimension
    size_t dim1() const 
    {
      return dim1(orientation());
    }

    // return minor dimension
    size_t dim2() const 
    {
      return dim2(orientation());
    }

    // Returns the row for row_major otherwise the column
    // Trailing _ due to conflicts with macro major
    size_t major_(size_t r, size_t c) const
    {
	return major_(r, c, orientation());
    }

    // Returns the row for col_major otherwise the column
    // Trailing _ for consistency with major
    size_t minor_(size_t r, size_t c) const
    {
	return major_(c, r, orientation());
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
    dim_type dimensions() const 
    {
      return dim; 
    }

    // returns n-th value in consecutive memory
    // (whatever this means in the corr. matrix format)
    value_type value_n(size_t offset) const 
    { 
      return data[offset]; 
    }
    
  };

}} // namespace mtl::detail

#endif // MTL_BASE_MATRIX_INCLUDE
