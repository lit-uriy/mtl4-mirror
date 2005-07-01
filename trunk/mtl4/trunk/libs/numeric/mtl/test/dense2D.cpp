// $COPYRIGHT$

#include <iostream>
#include <mtl/base_types.hpp>
#include <mtl/dense2D.hpp>

using namespace mtl;
using namespace std;

int main(int argc, char** argv) {
  double val[] = {1., 2., 3., 4., 5., 6.};
  dense2D<double, col_major> matrix(dim_type(2, 3), val, val+6);

  for (dense_el_cursor<double> cursor = matrix.ebegin(); cursor != matrix.eend(); ++cursor)
    cout << "matrix[" << row(matrix, *cursor) << ", " << col(matrix, *cursor)
	 << "] = " << value(matrix, *cursor) << endl;
  return 0;
} 
