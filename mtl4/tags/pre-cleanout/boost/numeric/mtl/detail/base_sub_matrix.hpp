// $COPYRIGHT$

#ifndef MTL_BASE_SUB_MATRIX_INCLUDE
#define MTL_BASE_SUB_MATRIX_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/index.hpp>
#include <boost/numeric/mtl/mtl_exception.hpp>

namespace mtl { namespace detail {

// Base class for sub-matrices
// Contains only very simple functionality that is used in all sub-matrices
// But also used in some complete matrices
template <class Elt, class Parameters>
struct base_sub_matrix 
{
    typedef Elt                               value_type;
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    static bool const                         on_stack= Parameters::on_stack;
    typedef std::size_t                       size_type;

  protected:
    size_type                       nnz,       // # of non-zeros, to be set by derived matrix (drop maybe?)
                                    my_begin_row, my_end_row,
                                    my_begin_col, my_end_col;

  public:
    base_sub_matrix() :  nnz(0), my_begin_row(0), my_end_row(0), my_begin_col(0), my_end_col(0) {}
   
    void set_ranges(size_type br, size_type er, size_type bc, size_type ec)
    {
	throw_debug_exception(br > er, "begin row > end row\n");
	throw_debug_exception(bc > ec, "begin column > end column\n");
	my_begin_row= br; my_end_row= er; my_begin_col= bc; my_end_col= ec;
    }

    explicit base_sub_matrix(size_type br, size_type er, size_type bc, size_type ec) : nnz(0)
    {
	set_ranges(br, er, bc, ec);
    }
 
    // Number of rows
    size_type num_rows() const 
    {
	return my_end_row - my_begin_row;
    }

    // First row taking indexing into account
    size_type begin_row() const 
    {
	return index::change_to(index_type(), my_begin_row);
    }
    
    // Past-end row taking indexing into account
    size_type end_row() const 
    {
	return index::change_to(index_type(), my_end_row);
    }

    // Number of columns
    size_type num_cols() const 
    {
	return my_end_col - my_begin_col;
    }

    // First column taking indexing into account
    size_type begin_col() const 
    {
	return index::change_to(index_type(), my_begin_col);
    }
    
    // Past-end column taking indexing into account
    size_type end_col() const 
    {
	return index::change_to(index_type(), my_end_col);
    }

  protected:
    // dispatched functions for major dimension
    size_type dim1(row_major) const 
    {
	return num_rows();
    }

    size_type dim1(col_major) const 
    {
	return num_cols();
    }

    // dispatched functions for minor dimension
    size_type dim2(row_major) const 
    {
	return num_cols();
    }

    size_type dim2(col_major) const 
    {
	return num_rows();
    }
  
    // Dispatched functions for major
    // Trailing _ due to conflicts with macro major
    size_type major_(size_type r, size_type, row_major) const
    {
	return r; 
    }

    size_type major_(size_type, size_type c, col_major) const
    {
	return c; 
    }    

  public:
    // return major dimension
    size_type dim1() const 
    {
	return dim1(orientation());
    }

    // return minor dimension
    size_type dim2() const 
    {
	return dim2(orientation());
    }

    // Returns the row for row_major otherwise the column
    // Trailing _ due to conflicts with macro major
    size_type major_(size_type r, size_type c) const
    {
	return major_(r, c, orientation());
    }

    // Returns the row for col_major otherwise the column
    // Trailing _ for consistency with major
    size_type minor_(size_type r, size_type c) const
    {
	return major_(c, r, orientation());
    }
};


}} // namespace mtl::detail

#endif // MTL_BASE_SUB_MATRIX_INCLUDE
