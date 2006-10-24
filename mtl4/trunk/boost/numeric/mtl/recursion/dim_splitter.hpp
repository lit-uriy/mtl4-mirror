// $COPYRIGHT$

#ifndef MTL_DIM_SPLITTER_INCLUDE
#define MTL_DIM_SPLITTER_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/recursion/utilities.hpp>

namespace mtl { namespace recursion {

// Splits dimensions of a matrix separately into a first part that
//   is the largest power of 2 smaller than m or n, plus rest;
//   doesn't yield empty submatrices if both dimension > 1
template <typename Matrix> 
struct separate_dim_splitter
{
    typedef typename Matrix::size_type                            size_type;

    explicit separate_dim_splitter(Matrix const& matrix) 
	: matrix(matrix),
	  my_row_split(matrix.begin_row() + first_part(matrix.num_rows())),
	  my_col_split(matrix.begin_col() + first_part(matrix.num_cols()))
    {}

    // End of northern half and beginning of southern
    size_type row_split() const
    {
	return my_row_split;
    }

    // End of western half and beginning of eastern
    size_type col_split() const
    {
	return my_col_split;
    }

private:
    Matrix const&   matrix;
    int             my_row_split, my_col_split;
};

// Splits dimensions of a matrix separately into a first part that
//   is the largest power of 2 smaller than the maximum of m and n;
//   can yield 1 or two empty submatrices if matrix is rather unproportional;
//   helps creating square submatrices
template <typename Matrix> 
struct max_dim_splitter
{
    typedef typename Matrix::size_type                            size_type;

    explicit max_dim_splitter(Matrix const& matrix) 
	: // matrix(matrix),
	  my_split(std::max(first_part(matrix.num_rows()), first_part(matrix.num_cols()))),
	  my_row_split(std::min(matrix.begin_row() + my_split, matrix.end_row())),
	  my_col_split(std::min(matrix.begin_col() + my_split, matrix.end_col()))
   {}

    // End of northern half and beginning of southern (limited to end_row)
    size_type row_split() const
    {
	return my_row_split;
    }

    // End of western half and beginning of eastern (limited to end_col)
    size_type col_split() const
    {
	return my_col_split;
    }

private:
    //    Matrix const&   matrix;
    int             my_split, // minimal 2^(k-1) such that 2^k >= max(num_rows, num_cols)
                    my_row_split, my_col_split;
};


// Splitting within bounding box of power of 2, using recurators
// For instance, the upper left part of a 530 x 17 matrix is
//   530 x 17   if the bound is 2048 or larger
//   512 x 17   if the bound is 1024
//                 bound of 512 or smaller is a wrong bound
template <typename Recurator>
struct outer_bound_splitter
{
    typedef typename Recurator::size_type                            size_type;

    explicit outer_bound_splitter(Recurator const& recurator) 
    {
	Matrix const& matrix= recurator.get_value();
	my_row_split= std::min(matrix.begin_row() + recurator.bound() / 2, matrix.end_row());
	my_col_split= std::min(matrix.begin_col() + recurator.bound() / 2, matrix.end_col());
    }


    // End of northern half and beginning of southern (limited to end_row)
    size_type row_split() const
    {
	return my_row_split;
    }

    // End of western half and beginning of eastern (limited to end_col)
    size_type col_split() const
    {
	return my_col_split;
    }

private:
    //    Matrix const&   matrix;
    int             my_split, // minimal 2^(k-1) such that 2^k >= max(num_rows, num_cols)
                    my_row_split, my_col_split;
};



}} // namespace mtl::recursion 

#endif // MTL_DIM_SPLITTER_INCLUDE
