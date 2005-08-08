// $COPYRIGHT$

#include <iostream>
#include <boost/tuple/tuple.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>


using namespace mtl;
using namespace std;

int main(int argc, char** argv) {
  typedef dense2D<double, col_major, mtl::index::f_index, mtl::fix_dim_type<2, 3> > matrix_type;
  double        val[] = {1., 2., 3., 4., 5., 6.};
  matrix_type   matrix(val, val+6);
//   typedef dense2D<double, col_major, mtl::index::f_index> matrix_type;
//   double        val[] = {1., 2., 3., 4., 5., 6.};
//   matrix_type   matrix(dim_type(2, 3), val, val+6);

  // traits::row<matrix_type>::type row(matrix);
  traits::row<matrix_type>::type r = row(matrix); // alternatively
  // traits::col<matrix_type>::type col(matrix);
  traits::col<matrix_type>::type c = col(matrix);
  // traits::value<matrix_type>::type value(matrix);
  traits::value<matrix_type>::type v = value(matrix);
  
  matrix_type::el_cursor_type cursor, end;
  for (boost::tie(cursor, end)= matrix.elements(); cursor != end; ++cursor)
    cout << "matrix[" << r(*cursor) << ", " << c(*cursor)
	 << "] = " << v(*cursor) << '\n';

  boost::tie(cursor, end)= matrix.elements();
  v(*cursor, 7.0);
  for (; cursor != end; ++cursor)
    cout << "matrix[" << r(*cursor) << ", " << c(*cursor)
	 << "] = " << v(*cursor) << '\n';

  cout << "matrix(1, 1) = " << matrix(1, 1) << '\n';
  cout << "matrix dimensions = " << matrix.dim_ref() << '\n';

  typedef transposed_view<matrix_type> trans_matrix_type;
  trans_matrix_type   trans_matrix(matrix);
  trans_matrix_type::el_cursor_type tcursor, tend;

  traits::row<trans_matrix_type>::type tr= row(trans_matrix);
  traits::col<trans_matrix_type>::type tc = col(trans_matrix);
  traits::value<trans_matrix_type>::type tv = value(trans_matrix);

  // value(matrix, *tcursor, 11.0);
  for (boost::tie(tcursor, tend)= trans_matrix.elements(); tcursor != tend; ++tcursor)
    cout << "trans_matrix[" << tr(*tcursor) << ", " << tc(*tcursor)
	 << "] = " << tv(*tcursor) << '\n';
  cout << "trans_matrix dimensions = " << trans_matrix.dim_ref() << '\n';

  return 0;
} 



// ==== old code, will be thrown away when new one works

//   matrix_type::el_cursor_type cursor, end;
//   for (boost::tie(cursor, end)= matrix.elements(); cursor != end; ++cursor)
//     cout << "matrix[" << row(matrix, *cursor) << ", " << col(matrix, *cursor)
// 	 << "] = " << value(matrix, *cursor) << '\n';

//   boost::tie(cursor, end)= matrix.elements();
//   value(matrix, *cursor, 7.0);
//   for (; cursor != end; ++cursor)
//     cout << "matrix[" << row(matrix, *cursor) << ", " << col(matrix, *cursor)
// 	 << "] = " << value(matrix, *cursor) << '\n';

//   cout << "matrix(1, 1) = " << matrix(1, 1) << '\n';
//   cout << "matrix dimensions = " << matrix.dim_ref() << '\n';

//   typedef transposed_view<matrix_type> trans_matrix_type;
//   trans_matrix_type   trans_matrix(matrix);
//   trans_matrix_type::el_cursor_type tcursor, tend;
//   boost::tie(tcursor, tend)= trans_matrix.elements();

//   // value(matrix, *tcursor, 11.0);
//   for (; tcursor != tend; ++tcursor)
//     cout << "trans_matrix[" << row(trans_matrix, *tcursor) << ", " << col(trans_matrix, *tcursor)
// 	 << "] = " << value(trans_matrix, *tcursor) << '\n';
//   cout << "trans_matrix dimensions = " << trans_matrix.dim_ref() << '\n';
