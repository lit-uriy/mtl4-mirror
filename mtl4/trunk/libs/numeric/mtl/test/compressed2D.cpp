// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/compressed2D.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>

using namespace mtl;
using namespace std;


int test_main(int argc, char* argv[])
{
    typedef matrix_parameters<row_major, mtl::index::c_index, fixed::dimensions<4, 5> > parameters1;
    typedef compressed2D<double, Parameters> matrix_type;
    matrix_type   matrix;
	
    size_t        sts[] = {0, 3, 3, 5, 10},
               	  ind[] = {1, 2, 3, 0, 4, 0, 1, 2, 3, 4};
    double        val[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    matrix.raw_copy(val, val+10, sts, ind);

    for (size_t r = 0; r < matrix.num_rows(); ++r)
	for (size_t c = 0; c < matrix.num_cols(); ++c)
	    cout << r << ", " << c << ": " << matrix.indexer(r, c);

    return 0;
}
