// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/morton_dense.hpp>
//#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>


using namespace mtl;
using namespace std;  


struct test_morton_dense
{
    template <typename Matrix, typename Tag>
    void two_d_iteration(char const* outer, Matrix & matrix, Tag)
    {
	typename traits::row<Matrix>::type                         row(matrix);
	typename traits::col<Matrix>::type                         col(matrix);
	typename traits::value<Matrix>::type                       value(matrix);
	typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;

	cout << outer << '\n';
	for (cursor_type cursor = begin<Tag>(matrix), cend = end<Tag>(matrix); cursor != cend; ++cursor) {
	    typedef glas::tags::all_t     inner_tag;
	    typedef typename traits::range_generator<inner_tag, cursor_type>::type icursor_type;
	    for (icursor_type icursor = begin<inner_tag>(cursor), icend = end<inner_tag>(cursor); icursor != icend; ++icursor)
		cout << "matrix[" << row(*icursor) << ", " << col(*icursor) << "] = " << value(*icursor) << '\n';
	}
    }

    template <typename Matrix>
    void one_d_iteration(char const* name, Matrix & matrix)
    {
	typename traits::row<Matrix>::type                         row(matrix);
	typename traits::col<Matrix>::type                         col(matrix);
	typename traits::value<Matrix>::type                       value(matrix);
	typedef  glas::tags::nz_t                                  tag;
	typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;

	cout << name << "\nElements: \n";
	for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor) {
	    cout << "matrix[" << row(*cursor) << ", " << col(*cursor) << "] = " << value(*cursor) << '\n';
	}
    }
    
    template <typename Matrix>
    void operator() (Matrix& matrix)
    {
	one_d_iteration("\nMatrix", matrix);
	two_d_iteration("\nRows: ", matrix, glas::tags::row_t());
	two_d_iteration("\nColumns: ", matrix, glas::tags::col_t());

	transposed_view<Matrix> trans_matrix(matrix);
	one_d_iteration("\nTransposed matrix", trans_matrix);
	two_d_iteration("\nRows: ", trans_matrix, glas::tags::row_t());
	two_d_iteration("\nColumns: ", trans_matrix, glas::tags::col_t());
    }
};


 
int test_main(int argc, char* argv[])
{
    typedef morton_dense<double,  0x55555555, matrix_parameters<> > matrix_type;    
    matrix_type matrix(non_fixed::dimensions(2, 3));
   
    traits::value<matrix_type>::type                       v = value(matrix);

    morton_dense_el_cursor<0x55555555>   cursor(0, 0, 3), cursor_end(2, 0, 3);
    for (double x= 7.3; cursor != cursor_end; ++cursor, x+= 1.0)
	v(cursor, x);

    test_morton_dense()(matrix);
    return 0;
}
