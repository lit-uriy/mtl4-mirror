// $COPYRIGHT$

#ifndef MTL_BASE_MATRIX_INCLUDE
#define MTL_BASE_MATRIX_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/index.hpp>

// temp solution for backward compatibility
#include <boost/numeric/mtl/detail/contiguous_memory_matrix.hpp>

namespace mtl { namespace detail {
using std::size_t;
  
// Base class for other matrices
// Contains only very simple functionality that is used in all matrices
template <class Elt, class Parameters>
struct base_matrix 
{
    typedef Elt                               value_type;
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    static bool const                         on_stack= Parameters::on_stack;
  protected:
    dim_type                        dim;       // # of rows and columns
    size_t                          nnz;       // # of non-zeros, to be set by derived matrix
    
  public:
    base_matrix() :  nnz(0) {}

    // setting dimension
    explicit base_matrix(mtl::non_fixed::dimensions d) : dim(d), nnz(0) {}
   
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
	
    // returns copy of dim
    dim_type dimensions() const 
    {
      return dim; 
    }

    
};


}} // namespace mtl::detail

#endif // MTL_BASE_MATRIX_INCLUDE
