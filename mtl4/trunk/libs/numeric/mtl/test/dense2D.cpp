// $COPYRIGHT$

#include <iostream>
#include <boost/tuple/tuple.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dense2D.hpp>
// #include <boost/numeric/mtl/transposed_view.hpp>

#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>

using namespace mtl;
using namespace std;
// using namespace glas::tags;

int main(int argc, char** argv) {
    typedef dense2D<double, col_major, mtl::index::f_index, mtl::fixed::dimensions<2, 3> > matrix_type;
    matrix_type   matrix;

    double        val[] = {1., 2., 3., 4., 5., 6.};
    raw_copy(val, val+6, matrix);

    traits::row<matrix_type>::type   r = row(matrix); 
    traits::col<matrix_type>::type   c = col(matrix);
    traits::value<matrix_type>::type v = value(matrix);
  
    typedef glas::tags::nz_t                                tag;
    typedef traits::range_generator<tag, matrix_type>::type cursor_type;
    for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor)
	cout << "matrix[" << r(*cursor) << ", " << c(*cursor) << "] = " << v(*cursor) << '\n';

#if 0

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

#endif

  return 0;
} 


