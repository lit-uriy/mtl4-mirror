// $COPYRIGHT$

#ifndef MTL_MATRIX_RECURATOR_INCLUDE
#define MTL_MATRIX_RECURATOR_INCLUDE

#include <boost/numeric/mtl/recursion/utilities.hpp>

namespace mtl { namespace recursion {

// To use matrix_recurator with const matrices Reference must be 'Matrix const&'
template <typename Matrix> //, typename Reference = Matrix&>
struct matrix_recurator
{
    typedef matrix_recurator            self;
    typedef Matrix                      matrix_type;
    typedef Matrix                      Reference;    // Hack !!!!!
    typedef typename Matrix::size_type  size_type;

    explicit matrix_recurator(Reference matrix) : matrix(&matrix) {}

    Reference get_value()
    {
	return *matrix;
    }

  protected:
    // End of northern half and beginning of southern
    size_type row_split() const
    {
	return matrix->begin_row() + first_part(matrix->end_row() - matrix->begin_row());
    }

    // End of western half and beginning of eastern
    size_type col_split() const
    {
	return matrix->begin_col() + first_part(matrix->end_col() - matrix->begin_col());
    }

  public:
    self north_west()
    {
	return self(matrix->sub_matrix(matrix->begin_row(), row_split(),
				       matrix->begin_col(), col_split()));
    }

  protected:
    Reference*   matrix;
};


}} // namespace mtl::recursion

#endif // MTL_MATRIX_RECURATOR_INCLUDE
