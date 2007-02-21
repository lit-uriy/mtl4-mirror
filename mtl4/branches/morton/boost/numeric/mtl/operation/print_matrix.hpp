// $COPYRIGHT$

#ifndef MTL_PRINT_MATRIX_INCLUDE
#define MTL_PRINT_MATRIX_INCLUDE

#include <iostream>
#include <boost/numeric/mtl/utility/glas_tag.hpp>
#include <boost/numeric/mtl/utility/traits.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>

namespace mtl {

template <typename Matrix>
std::ostream& print_matrix(Matrix const& matrix, std::ostream& out= std::cout, int width= 3, int precision= 2)
{
    for (size_t r = matrix.begin_row(); r < matrix.end_row(); ++r) {
	out << '[';
	for (size_t c = matrix.begin_col(); c < matrix.end_col(); ++c) {
	    out.fill (' '); out.width (width); out.precision (precision); // out.flags (std::ios_base::right);
	    if (precision)
		out.precision(precision); 
	    out << matrix(r, c) 
		<< (c < matrix.end_col() - 1 ? " " : "]\n");
	}
    }
    return out;
}

// Deprecated 
template <typename Matrix>
void print_matrix_row_cursor(Matrix const& matrix, std::ostream& out= std::cout)
{
    typedef glas::tag::row                                          Tag;
    typename traits::const_value<Matrix>::type                         value(matrix);
    typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;

    for (cursor_type cursor = begin<Tag>(matrix), cend = end<Tag>(matrix); cursor != cend; ++cursor) {
	out << '[';
	typedef glas::tag::all     inner_tag;
	typedef typename traits::range_generator<inner_tag, cursor_type>::type icursor_type;
	for (icursor_type icursor = begin<inner_tag>(cursor), icend = end<inner_tag>(cursor); icursor != icend; ) {
	    out << value(*icursor);
	    ++icursor;
	    out << ( icursor != icend ? ", " : "]\n");
	}
    }
}

} // namespace mtl

#endif // MTL_PRINT_MATRIX_INCLUDE
