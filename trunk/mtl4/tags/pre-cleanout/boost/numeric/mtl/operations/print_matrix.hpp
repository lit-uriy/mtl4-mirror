// $COPYRIGHT$

#ifndef MTL_PRINT_MATRIX_INCLUDE
#define MTL_PRINT_MATRIX_INCLUDE

#include <iostream>

namespace mtl {

template <typename Matrix>
void print_matrix(Matrix const& matrix, std::ostream& out= std::cout)
{
    for (size_t r = matrix.begin_row(); r < matrix.end_row(); ++r) {
	out << '[';
	for (size_t c = matrix.begin_col(); c < matrix.end_col(); ++c) {
	    out << matrix(r, c) 
		<< (c < matrix.end_col() - 1 ? ", " : "]\n"); } 
    }    
}

} // namespace mtl

#endif // MTL_PRINT_MATRIX_INCLUDE
