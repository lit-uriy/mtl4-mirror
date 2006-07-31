// $COPYRIGHT$

#ifndef MTL_MATRIX_RECURATOR_INCLUDE
#define MTL_MATRIX_RECURATOR_INCLUDE

#include <boost/numeric/mtl/recursion/utilities.hpp>

namespace mtl { namespace recursion {

// To use matrix_recurator with const matrices Reference must be 'Matrix const&'
template <typename Matrix> //, typename Reference = Matrix&>
struct matrix_recurator
{
    typedef matrix_recurator                    self;
    typedef Matrix                              matrix_type;
    typedef typename Matrix::sub_matrix_type    sub_matrix_type;
    typedef typename Matrix::size_type          size_type;

    // Constructor takes the whole matrix as sub-matrix
    // This allows to have different type for the matrix and the sub-matrix
    // This also enables matrices to have references as sub-matrices
    explicit matrix_recurator(Matrix& matrix) 
      : my_sub_matrix(matrix.sub_matrix(matrix.begin_row(), matrix.end_row(),
					matrix.begin_col(), matrix.end_col()))
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
    self north_west()
    {
	sub_matrix_type sm(my_sub_matrix.sub_matrix(my_sub_matrix.begin_row(), row_split(),
							  my_sub_matrix.begin_col(), col_split()));
	self tmp(sm);
	return tmp;

	// return self(my_sub_matrix.sub_matrix(my_sub_matrix.begin_row(), row_split(),
	//			     my_sub_matrix.begin_col(), col_split()));
    }

    self south_west()
    {
	sub_matrix_type sm(my_sub_matrix.sub_matrix(row_split(), my_sub_matrix.end_row(), 
						    my_sub_matrix.begin_col(), col_split()));
	self tmp(sm);
	return tmp;
    }

    self north_east()
    {
	sub_matrix_type sm(my_sub_matrix.sub_matrix(my_sub_matrix.begin_row(), row_split(),
						    col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    self south_east()
    {
	sub_matrix_type sm(my_sub_matrix.sub_matrix(row_split(), my_sub_matrix.end_row(), 
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
