// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>

using namespace mtl;
using namespace std;
 
 
int test_main(int argc, char* argv[])
{

    typedef morton_dense<double, matrix_parameters<> > matrix_type;
    
    matrix_type matrix(non_fixed::dimensions(2, 3));
   
#if 0
    traits::row<matrix_type>::type                         r = row(matrix);
    traits::col<matrix_type>::type                         c = col(matrix);
    traits::value<matrix_type>::type                       v = value(matrix);

    
    typedef glas::tags::all_t all_tag;
    typedef traits::range_generator<glas::tags::all_tag, matrix_type> cursor_type;
    for (cursor_type cursor = begin<all_tag>(matrix), cend = end<all_tag>(matrix); cursor != cend; ++cursor) 
	cout << "matrix[" << r(*cursor) << ", " << c(*cursor) << "] = " << v(*cursor) << '\n';

    typedef glas::tags::row_t row_tag;
    typedef traits::range_generator<row_tag, matrix_type> row_cursor_type;
    for (row_cursor_type cursor = begin<row_tag>(matrix), cend = end<row_tag>(matrix); cursor != cend; ++cursor) {
	typedef typename traits::range_generator<all_tag, row_cursor_type>::type icursor_type;
	for (icursor_type icursor = begin<all_tag>(cursor), icend = end<all_tag>(cursor); icursor != icend; ++icursor)
		cout << "matrix[" << r(*icursor) << ", " << c(*icursor) << "] = " << v(*icursor) << '\n';
    }
	
    typedef glas::tags::col_t col_tag;
    typedef traits::range_generator<col_tag, matrix_type> col_cursor_type;
    for (col_cursor_type cursor = begin<col_tag>(matrix), cend = end<col_tag>(matrix); cursor != cend; ++cursor) {
	typedef typename traits::range_generator<all_tag, col_cursor_type>::type icursor_type;
	for (icursor_type icursor = begin<all_tag>(cursor), icend = end<all_tag>(cursor); icursor != icend; ++icursor)
		cout << "matrix[" << r(*icursor) << ", " << c(*icursor) << "] = " << v(*icursor) << '\n';
    }
#endif	

    return 0;
}
