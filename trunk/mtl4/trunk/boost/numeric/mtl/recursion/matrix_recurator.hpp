// $COPYRIGHT$

#ifndef MTL_MATRIX_RECURATOR_INCLUDE
#define MTL_MATRIX_RECURATOR_INCLUDE

#include <cmath>
#include <boost/numeric/mtl/recursion/utilities.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/recursion/dim_splitter.hpp>

// supersedes version in trunk (will be deleted one day)

namespace mtl { namespace recursion {


template <typename Recurator1, typename Recurator2>
void inline equalize_depth(Recurator1& r1, Recurator2& r2);

template <typename Recurator1, typename Recurator2, typename Recurator3>
void inline equalize_depth(Recurator1& r1, Recurator2& r2, Recurator3& r3);


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
    
    // template <typename Matrix> why was it templated ???
    sub_matrix_type constructor_helper(Matrix const& matrix)
    {
	return sub_matrix(matrix, matrix.begin_row(), matrix.end_row(),
			  matrix.begin_col(), matrix.end_col());
    }

    // For views without own data, we need to generate a new sub_matrix as shared_ptr
    // template <typename Matrix>
    sub_matrix_type constructor_helper(transposed_view<Matrix> const& matrix)
    {
	typedef typename sub_matrix_t<Matrix>::sub_matrix_type   ref_sub_type;
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
    explicit matrix_recurator(Matrix const& matrix, size_type bound= 0) 
	: my_sub_matrix(constructor_helper(matrix)), my_bound(outer_bound(matrix)),
	  my_first_row(0), my_first_col(0)         // splitter(*this)
    {
      if (bound == 0)
	my_bound= outer_bound(matrix);
      else {
	assert(is_power_of_2(bound));
	assert(bound >= matrix.num_rows() && bound >= matrix.num_cols());
	my_bound= bound;
      }
    }

    // Sub-matrices are copied directly
    // explicit matrix_recurator(sub_matrix_type sub_matrix) : my_sub_matrix(sub_matrix) {}
    
#if 0
    sub_matrix_type& get_value()
    {
	return my_sub_matrix;
    }
#endif

    sub_matrix_type get_value() const
    {
	using std::min;
	size_type begin_row= my_sub_matrix.begin_row() + my_first_row,
	          end_row= min(begin_row + my_bound, my_sub_matrix.end_row()),
	          begin_col= my_sub_matrix.begin_col() + my_first_col,
	          end_col= min(begin_col + my_bound, my_sub_matrix.end_col());
	return sub_matrix(my_sub_matrix, begin_row, end_row, begin_col, end_col);
    }


    // Returning quadrants for non-const recurator

    self north_west() const
    {
	self tmp(*this);
	tmp.my_bound >>= 1; // divide by 2
	return tmp;
    }

    self south_west() const
    {
	self tmp(*this);
	tmp.my_bound >>= 1; // divide by 2
	tmp.my_first_row += tmp.my_bound;
	return tmp;
    }

    self north_east() const
    {
	self tmp(*this);
	tmp.my_bound >>= 1; // divide by 2
	tmp.my_first_col += tmp.my_bound;
	return tmp;
    }

    self south_east() const
    {
	self tmp(*this);
	tmp.my_bound >>= 1; // divide by 2
	tmp.my_first_row += tmp.my_bound;
	tmp.my_first_col += tmp.my_bound;
	return tmp;
    }

    // Checking whether a quadrant is empty
    // Generation of recurator is fast enough
    // r.south_east().empty() shouldn't be much slower than r.south_east_empty()

    // For completeness
    bool north_west_empty() const
    {
	return false;
    }

    bool north_east_empty() const
    {
	return my_first_row >= my_sub_matrix.num_rows() || my_first_col+my_bound/2 >= my_sub_matrix.num_cols();
    }

    bool south_west_empty() const
    {
	return my_first_row+my_bound/2 >= my_sub_matrix.num_rows() || my_first_col >= my_sub_matrix.num_cols();
    }

    bool south_east_empty() const
    {
	return my_first_row+my_bound/2 >= my_sub_matrix.num_rows() || my_first_col+my_bound/2 >= my_sub_matrix.num_cols();
    }

    bool is_empty() const
    {
	return my_first_row >= my_sub_matrix.num_rows() || my_first_col >= my_sub_matrix.num_cols();
    }


    size_type bound() const
    {
	// assert(my_bound >= my_sub_matrix.num_rows() && my_bound >= my_sub_matrix.num_cols());
	return my_bound;
    }

    template <typename R1, typename R2> friend void equalize_depth (R1&, R2&);   
    template <typename R1, typename R2, typename R3> friend void equalize_depth (R1&, R2&, R3&);

  protected:
    sub_matrix_type     my_sub_matrix;
    size_type           my_bound, // virtual matrix size, upper bound of current sub-matrix
	                my_first_row, my_first_col; // first entry in submatrix (w.r.t. 0-indexing)

    // splitter_type       splitter;
};



// To use matrix_recurator with const matrices Reference must be 'Matrix const&'
template <typename Matrix, typename Splitter = max_dim_splitter<Matrix> >
struct matrix_recurator_s
{
    typedef matrix_recurator_s                                    self;
    typedef Matrix                                                matrix_type;
    typedef Splitter                                              splitter_type;
    typedef typename sub_matrix_t<Matrix>::sub_matrix_type        sub_matrix_type;
    typedef typename sub_matrix_t<Matrix>::const_sub_matrix_type  const_sub_matrix_type;
    typedef typename Matrix::size_type                            size_type;
    // typedef outer_bound_splitter<self>                            splitter_type;

private:
    
    // template <typename Matrix> why was it templated ???
    sub_matrix_type constructor_helper(Matrix const& matrix)
    {
	return sub_matrix(matrix, matrix.begin_row(), matrix.end_row(),
			  matrix.begin_col(), matrix.end_col());
    }

    // For views without own data, we need to generate a new sub_matrix as shared_ptr
    // template <typename Matrix>
    sub_matrix_type constructor_helper(transposed_view<Matrix> const& matrix)
    {
	typedef typename sub_matrix_t<Matrix>::sub_matrix_type   ref_sub_type;
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
    explicit matrix_recurator_s(Matrix const& matrix, size_type bound= 0) 
	: my_sub_matrix(constructor_helper(matrix)), my_bound(outer_bound(matrix)),
	  splitter(my_sub_matrix)
    {
      if (bound == 0)
	my_bound= outer_bound(matrix);
      else {
	assert(is_power_of_2(bound));
	assert(bound >= matrix.num_rows() && bound >= matrix.num_cols());
	my_bound= bound;
      }
    }

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
	self tmp(sm, my_bound / 2);
	return tmp;
    }

    self south_west()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      my_sub_matrix.begin_col(), splitter.col_split()));
	self tmp(sm, my_bound / 2);
	return tmp;
    }

    self north_east()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, my_sub_matrix.begin_row(), splitter.row_split(),
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm, my_bound / 2);
	return tmp;
    }

    self south_east()
    {
	sub_matrix_type sm(sub_matrix(my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm, my_bound / 2);
	return tmp;
    }

    // Returning quadrants for const recurator

    self const north_west() const
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, my_sub_matrix.begin_row(), splitter.row_split(),
				      my_sub_matrix.begin_col(), splitter.col_split()));
	self tmp(sm, my_bound / 2);
	return tmp;
    }

    self const south_west() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      my_sub_matrix.begin_col(), splitter.col_split()));
	self tmp(sm, my_bound / 2);
	return tmp;
    }

    self const north_east() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, my_sub_matrix.begin_row(), splitter.row_split(),
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm, my_bound / 2);
	return tmp;
    }

    self const south_east() const 
    {
	sub_matrix_type sm(sub_matrix(const_cast<self*>(this)->my_sub_matrix, splitter.row_split(), my_sub_matrix.end_row(), 
				      splitter.col_split(), my_sub_matrix.end_col()));
	self tmp(sm, my_bound / 2);
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

    bool is_empty() const
    {
	return my_sub_matrix.begin_row() == my_sub_matrix.end_row()
	       || my_sub_matrix.begin_col() == my_sub_matrix.end_col();
    }

#if 0
    bool is_leaf() const
    {
	return my_sub_matrix.num_rows() < 2 || my_sub_matrix.num_cols() < 2;
    }
#endif

    size_type bound() const
    {
	assert(my_bound >= my_sub_matrix.num_rows() && my_bound >= my_sub_matrix.num_cols());
	return my_bound;
    }

    template <typename R1, typename R2> friend void equalize_depth (R1&, R2&);   
    template <typename R1, typename R2, typename R3> friend void equalize_depth (R1&, R2&, R3&);

  protected:
    sub_matrix_type     my_sub_matrix;
    size_type           my_bound;
    splitter_type       splitter;
};


template <typename Recurator1, typename Recurator2>
void inline equalize_depth(Recurator1& r1, Recurator2& r2)
{
    typename Recurator1::size_type max_bound= std::max(r1.bound(), r2.bound());
    r1.my_bound= max_bound;
    r2.my_bound= max_bound;
}

template <typename Recurator1, typename Recurator2, typename Recurator3>
void inline equalize_depth(Recurator1& r1, Recurator2& r2, Recurator3& r3)
{
    typename Recurator1::size_type max_bound= std::max(std::max(r1.bound(), r2.bound()), r3.bound());
    r1.my_bound= max_bound;
    r2.my_bound= max_bound;
    r3.my_bound= max_bound;
}


} // namespace recursion

using recursion::matrix_recurator;

} // namespace mtl

#endif // MTL_MATRIX_RECURATOR_INCLUDE
