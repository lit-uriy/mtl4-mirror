// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/compressed2D.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/maybe.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>

using namespace mtl;
using namespace std;


int test_main(int argc, char* argv[])
{
    typedef matrix_parameters<row_major, mtl::index::c_index, non_fixed::dimensions > parameters;
    typedef compressed2D<double, parameters> matrix_type;
    matrix_type   matrix(non_fixed::dimensions(8, 6)); 
	
    size_t        sts[] = {0, 2, 3, 7, 9, 12, 16, 16, 18},
                  ind[] = {1, 4, 2, 0, 1, 2, 3, 3, 5, 1, 3, 4, 1, 3, 4, 5, 2, 5};
    double        val[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.};
    matrix.raw_copy(val, val+18, sts, ind);

    compressed2D_inserter<double, parameters> inserter(matrix, 3);    
 
#if 0
    for (size_t r = 0; r < matrix.num_rows(); ++r)
	for (size_t c = 0; c < matrix.num_cols(); ++c)
	    cout << r << ", " << c << ": " << matrix.indexer(matrix, r, c) << '\n';

    for (size_t r = 0; r < matrix.num_rows(); ++r) {
	cout << '[';
	for (size_t c = 0; c < matrix.num_cols(); ++c) {
	    maybe<size_t> m = matrix.indexer(matrix, r, c);
	    cout << (m ? matrix.value_n(m.value()) : 0)
		 << (c < matrix.num_cols() - 1 ? ", " : "]\n"; } 
    }
#endif
    return 0;
}
 
