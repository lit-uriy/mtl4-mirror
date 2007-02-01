// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp> 
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>


using namespace mtl;
using namespace std;  


struct test_morton_dense 
{
    template <typename Matrix, typename Tag>
    void two_d_iteration(char const* outer, Matrix & matrix, Tag)
    {
	typename traits::row<Matrix>::type                                 row(matrix);
	typename traits::col<Matrix>::type                                 col(matrix);
	typename traits::value<Matrix>::type                               value(matrix);
	typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;

	cout << outer << '\n';
	for (cursor_type cursor = begin<Tag>(matrix), cend = end<Tag>(matrix); cursor != cend; ++cursor) {
	    typedef glas::tags::all_t     inner_tag;
	    typedef typename traits::range_generator<inner_tag, cursor_type>::type icursor_type;
	    for (icursor_type icursor = begin<inner_tag>(cursor), icend = end<inner_tag>(cursor); icursor != icend; ++icursor)
		cout << "matrix[" << row(*icursor) << ", " << col(*icursor) << "] = " << value(*icursor) << '\n';
	    icursor_type ibeg = begin<inner_tag>(cursor), icursor= ibeg + 2;
	    cout << "--\nmatrix[" << row(*icursor) << ", " << col(*icursor) << "] = " << value(*icursor) << "\n--\n";
	}
    }

    template <typename Matrix, typename Tag>
    void two_d_iterator_iteration(char const* outer, Matrix & matrix, Tag)
    {
	typename traits::row<Matrix>::type                                 row(matrix);
	typename traits::col<Matrix>::type                                 col(matrix);
	typename traits::value<Matrix>::type                               value(matrix);
	typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;

	cout << outer << '\n';
	for (cursor_type cursor = begin<Tag>(matrix), cend = end<Tag>(matrix); cursor != cend; ++cursor) {
	    typedef glas::tags::all_it     inner_tag;
	    typedef typename traits::range_generator<inner_tag, cursor_type>::type icursor_type;
	    for (icursor_type icursor = begin<inner_tag>(cursor), icend = end<inner_tag>(cursor); icursor != icend; ++icursor)
		cout << *icursor <<'\n';
	}
    } 

    template <typename Matrix> 
    void one_d_iteration(char const* name, Matrix & matrix)
    {
	typename traits::row<Matrix>::type                                 row(matrix);
	typename traits::col<Matrix>::type                                 col(matrix);
	typename traits::value<Matrix>::type                               value(matrix);
	typedef  glas::tags::nz_t                                          tag; 
	typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;

	cout << name << "\nElements: \n";
	for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor) {
	    cout << "matrix[" << row(*cursor) << ", " << col(*cursor) << "] = " << value(*cursor) << '\n';
	}
    }
    
    template <typename Matrix>
    void fill_matrix(Matrix & matrix)
    {
	typename traits::value<Matrix>::type                               value(matrix);
	typedef  glas::tags::nz_t                                          tag;
	typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;

	typename Matrix::value_type  v= 1;

	for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor) {
	    value(*cursor, v);
	    v+= 1;
	}
    }

    template <typename Matrix>
    void check_cursor_increment(Matrix& matrix)
    {
	typename traits::row<Matrix>::type                                 row(matrix);
	typename traits::col<Matrix>::type                                 col(matrix);
	typename traits::value<Matrix>::type                               value(matrix);
	typedef  glas::tags::nz_t                                          tag;
	typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;
	
	cursor_type cursor = begin<tag>(matrix);
	cout << "begin: matrix[" << row(*cursor) << ", " << col(*cursor) << "] = " << value(*cursor) << '\n';
	cursor.advance(2, 2);
	cout << "advance (2,2): matrix[" << row(*cursor) << ", " << col(*cursor) << "] = " << value(*cursor) << '\n';
	cursor.advance(-1, -1);
	cout << "advance (-1, -1): matrix[" << row(*cursor) << ", " << col(*cursor) << "] = " << value(*cursor) << '\n';
    }

    template <typename Matrix>
    void operator() (Matrix& matrix)
    {
	fill_matrix(matrix);
	check_cursor_increment(matrix);

	one_d_iteration("\nMatrix", matrix);
	two_d_iteration("\nRows: ", matrix, glas::tags::row_t());
	two_d_iteration("\nColumns: ", matrix, glas::tags::col_t());
	two_d_iterator_iteration("\nRows (iterator): ", matrix, glas::tags::row_t());
	two_d_iterator_iteration("\nColumns (iterator): ", matrix, glas::tags::col_t());

	transposed_view<Matrix> trans_matrix(matrix);
	one_d_iteration("\nTransposed matrix", trans_matrix);
	two_d_iteration("\nRows: ", trans_matrix, glas::tags::row_t());
	two_d_iteration("\nColumns: ", trans_matrix, glas::tags::col_t());
	two_d_iterator_iteration("\nRows (iterator): ", trans_matrix, glas::tags::row_t());
	two_d_iterator_iteration("\nColumns (iterator): ", trans_matrix, glas::tags::col_t());
    }
};



 
int test_main(int argc, char* argv[])
{
    morton_dense<double,  0x55555555> matrix1(3, 5);
    matrix1[1][3]= 2.3;
    cout << "matrix1[1][3] = " << matrix1[1][3] << endl;

    typedef morton_dense<int,  0x55555553> matrix2_type;
    matrix2_type                           matrix2(5, 6);
    matrix2[1][3]= 3;
    cout << "matrix2[1][3] = " << matrix2[1][3] << endl;
    

    test_morton_dense()(matrix1);
    test_morton_dense()(matrix2);

    return 0;

    typedef morton_dense<double,  0x55555555, matrix_parameters<> > matrix_type;    
    matrix_type matrix(non_fixed::dimensions(2, 3));
   
    traits::value<matrix_type>::type                       value(matrix);
 
    morton_dense_el_cursor<0x55555555>   cursor(0, 0, 3), cursor_end(2, 0, 3);
    for (double x= 7.3; cursor != cursor_end; ++cursor, x+= 1.0)
	value(cursor, x);

    test_morton_dense()(matrix);
    return 0;
}
