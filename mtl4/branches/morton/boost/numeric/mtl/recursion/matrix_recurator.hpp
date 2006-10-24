// $COPYRIGHT$

#ifndef MTL_MATRIX_RECURATOR_INCLUDE
#define MTL_MATRIX_RECURATOR_INCLUDE

#include <boost/numeric/mtl/recursion/utilities.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/recursion/dim_splitter.hpp>

// supersedes version in trunk (will be deleted one day)

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
    typedef outer_bound_splitter<self>                            splitter_type;

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
	typedef typename sub_matrix_t<M>::sub_matrix_type        ref_sub_type;
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
	: my_sub_matrix(constructor_helper(matrix)), my_bound(outer_bound(matrix)),
	  splitter(*this)
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

    // Returning quadrants for non-const recurator

    self north_west()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, my_sub_matrix.begin_row(), splitter.row_split(),
				      my_sub_matrix.begin_col(), splitter.col_split()));
	self tmp(sm);
	return tmp;
    }

    self south_west()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      my_sub_matrix.begin_col(), splitter.col_split()));
	self tmp(sm);
	return tmp;
    }

    self north_east()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, my_sub_matrix.begin_row(), splitter.row_split(),
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    self south_east()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    // Returning quadrants for const recurator

    self const north_west() const
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, my_sub_matrix.begin_row(), splitter.row_split(),
				      my_sub_matrix.begin_col(), splitter.col_split()));
	self tmp(sm);
	return tmp;
    }

    self const south_west() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      my_sub_matrix.begin_col(), splitter.col_split()));
	self tmp(sm);
	return tmp;
    }

    self const north_east() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, my_sub_matrix.begin_row(), splitter.row_split(),
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    self const south_east() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm);
	return tmp;
    }

    // Checking whether a quadrant is empty

    // For completeness
    bool north_west_empty() const
    {
	return false;
    }

    bool north_east_empty() const
    {
	return splitter.col_split() == my_sub_matrix.end_col();
    }

    bool south_west_empty() const
    {
	return splitter.row_split() == my_sub_matrix.end_row();
    }

    bool south_east_empty() const
    {
	return splitter.row_split() == my_sub_matrix.end_row() 
	       || splitter.col_split() == my_sub_matrix.end_col();
    }


    bool is_leaf() const
    {
	return my_sub_matrix.num_rows() < 2 || my_sub_matrix.num_cols() < 2;
    }

    size_type bound() const
    {
	assert(my_bound >= my_sub_matrix.num_rows() && my_bound >= my_sub_matrix.num_cols());
	return my_bound;
    }

  protected:
    sub_matrix_type     my_sub_matrix;
    size_type           my_bound;
    splitter_type       splitter;
};


}} // namespace mtl::recursion

#endif // MTL_MATRIX_RECURATOR_INCLUDE
