// $COPYRIGHT$

#include <iostream>
#include <mtl/base_types.hpp>
#include <mtl/fractalu.hpp>
#include <mtl/transposed_view.hpp>
#include <mtl/property_map.hpp>

using namespace mtl;
using namespace std;

int main(int argc, char** argv) {
  typedef fractalu<double, 3> matrix_type;
  matrix_type   matrix(dim_type(15, 15), 2.0);

//   matrix_type::el_cursor_type cursor= matrix.ebegin(), end= matrix.eend();
//   for (; cursor != end; ++cursor)
//     cout << "matrix[" << matrix.offset(*cursor) << "] = " 
// 	 << "matrix[" << row(matrix, *cursor) << ", " << col(matrix, *cursor)
// 	 << "] = " << value(matrix, *cursor) << endl;

//   typedef transposed_view<matrix_type> trans_matrix_type;
//   trans_matrix_type   trans_matrix(matrix);
//   trans_matrix_type::el_cursor_type tcursor= matrix.ebegin(), tend= matrix.eend();

//   for (; tcursor != tend; ++tcursor) 
//     cout << "trans_matrix[" << trans_matrix.offset(*tcursor) << "] = " 
// 	 << "trans_matrix[" << row(trans_matrix, *tcursor) << ", " << col(trans_matrix, *tcursor)
// 	 << "] = " << value(trans_matrix, *tcursor) << endl;
//   cout << "trans_matrix dimensions = " << trans_matrix.dim_ref() << endl;

  matrix_type::block_cursor_type bcursor= matrix.bbegin(), bend= matrix.bend();
  cout << "bend: = bcursor(" << bend.get_n() << ") = matrix ["  
       << row(matrix, *bend) << ", " << col(matrix, *bend) << "]" << endl;
  for (; bcursor != bend; ++bcursor)
    cout << "cursor(" << bcursor.get_n() << ") = "
	 << "matrix[" << row(matrix, *bcursor) << ", " << col(matrix, *bcursor)
	 << "] = " << value(matrix, *bcursor) 
	 << " // |" << bcursor.get_a() << 'x' << bcursor.get_b() << '|' << endl;
  return 0;
} 
