// $COPYRIGHT$

#include <iostream>
#include <mtl/base_types.hpp>
#include <mtl/dense2D.hpp>
#include <mtl/transposed_view.hpp>
#include <mtl/property_map.hpp>

using namespace mtl;
using namespace std;

int main(int argc, char** argv) {
  typedef dense2D<double, col_major, f_index> matrix_type;
  double        val[] = {1., 2., 3., 4., 5., 6.};
  matrix_type   matrix(dim_type(2, 3), val, val+6);

  matrix_type::el_cursor_type cursor= matrix.ebegin(), end= matrix.eend();
  for (; cursor != end; ++cursor)
    cout << "matrix[" << row(matrix, *cursor) << ", " << col(matrix, *cursor)
	 << "] = " << value(matrix, *cursor) << endl;

  cursor= matrix.ebegin();
  value(matrix, *cursor, 7.0);
  for (; cursor != end; ++cursor)
    cout << "matrix[" << row(matrix, *cursor) << ", " << col(matrix, *cursor)
	 << "] = " << value(matrix, *cursor) << endl;

  cout << "matrix(1, 1) = " << matrix(1, 1) << endl;
  cout << "matrix dimensions = " << matrix.dim_ref() << endl;

  typedef transposed_view<matrix_type> trans_matrix_type;
  trans_matrix_type   trans_matrix(matrix);
  trans_matrix_type::el_cursor_type tcursor= matrix.ebegin(), tend= matrix.eend();

  value(matrix, *tcursor, 11.0);
  for (; tcursor != tend; ++tcursor)
    cout << "trans_matrix[" << row(trans_matrix, *tcursor) << ", " << col(trans_matrix, *tcursor)
	 << "] = " << value(trans_matrix, *tcursor) << endl;
  cout << "trans_matrix dimensions = " << trans_matrix.dim_ref() << endl;

  return 0;
} 
