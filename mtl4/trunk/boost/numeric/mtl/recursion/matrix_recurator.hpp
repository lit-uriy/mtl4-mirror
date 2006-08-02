// $COPYRIGHT$

#ifndef MTL_MATRIX_RECURATOR_INCLUDE
#define MTL_MATRIX_RECURATOR_INCLUDE

#include <boost/numeric/mtl/recursion/utilities.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>

namespace mtl { namespace recursion {

// To use matrix_recurator with const matrices Reference must be 'Matrix const&'
template <typename Matrix> 
struct matrix_recurator
{
    typedef matrix_recurator                                      self;
    typedef Matrix                                                matrix_type;
    typedef typename sub_matrix_t<Matrix>::sub_matrix_type        sub_matrix_type;
    typedef typename sub_matrix_t<Matrix>::const_sub_matrix_type  const_sub_matrix_type;
    typedef typename Matrix::size_type                            size_type;

private:
    
    template <typename M>
    sub_matrix_type constructor_helper(M const& matrix)
    {
	return sub_matrix(matrix, matrix.begin_row(), matrix.end_row(),
			  matrix.begin_col(), matrix.end_col());
    }

    // For views without own data, we need to generate a new sub_matrix as shared_ptr
    template <typename M>
    sub_matrix_type constructor_helper(transposed_view<M> const& matrix)
    {
	typedef typename sub_matrix_t<M>::sub_matrix_type   ref_sub_type;
	typedef boost::shared_ptr<ref_sub_type>                  pointer_type;

	// Submatrix of referred matrix, colums and rows interchanged
	// Create a submatrix, whos address will be kept by transposed_view
	pointer_type p(new ref_sub_type(sub_matrix(matrix.ref, matrix.begin_col(), matrix.end_col(), 
						   matrix.begin_row(), matrix.end_row())));
	return sub_matrix_type(p); 
    }
    

public:
    // Constructor takes the whole matrix as sub-matrix
    // This allows to have different type for the matrix and the sub-matrix
    // This also enables matrices to have references as sub-matrices
    explicit matrix_recurator(Matrix const& matrix) 
	: my_sub_matrix(constructor_helper(matrix))
    {}

    // Sub-matrices are copied directly
    // explicit matrix_recurator(sub_matrix_type sub_matrix) : my_sub_matrix(sub_matrix) {}
    
    sub_matrix_type& get_value()
    {
	return my_sub_matrix;
    }

    sub_matrix_type const& get_value() const
    {
	return my_sub_matrix;
    }

  protected:
    // End of northern half and beginning of southern
    size_type row_split() const
    {
	return my_sub_matrix.begin_row() + first_part(my_sub_matrix.num_rows());
    }

    // End of western half and beginning of eastern
    size_type col_split() const
    {
	return my_sub_matrix.begin_col() + first_part(my_sub_matrix.num_cols());
    }

  public:


    // Returning quadrants for const recurator

    self north_west()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, my_sub_matrix.begin_row(), row_split(),
				      my_sub_matrix.begin_col(), col_split()));
	self tmp(sm);
	return tmp;
    }

    self south_west()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, row_split(), my_sub_matrix.end_row(), 
				      my_sub_matrix.begin_col(), col_split()));
	self tmp(sm);
	return tmp;
    }

    self north_east()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, my_sub_matrix.begin_row(), row_split(),
				      col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    self south_east()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, row_split(), my_sub_matrix.end_row(), 
				      col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    // Returning quadrants for const recurator

    self const north_west() const
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, my_sub_matrix.begin_row(), row_split(),
				      my_sub_matrix.begin_col(), col_split()));
	self tmp(sm);
	return tmp;
    }

    self const south_west() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, row_split(), my_sub_matrix.end_row(), 
				      my_sub_matrix.begin_col(), col_split()));
	self tmp(sm);
	return tmp;
    }

    self const north_east() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, my_sub_matrix.begin_row(), row_split(),
				      col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    self const south_east() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, row_split(), my_sub_matrix.end_row(), 
				      col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    bool is_leaf() const
    {
	return my_sub_matrix.num_rows() < 2 || my_sub_matrix.num_cols() < 2;
    }

  protected:
    sub_matrix_type     my_sub_matrix;
};


}} // namespace mtl::recursion

#endif // MTL_MATRIX_RECURATOR_INCLUDE
