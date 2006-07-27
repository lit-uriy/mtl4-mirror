// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>


using namespace mtl;
using namespace std;  


struct test_morton_dense 
{    
    template <typename Matrix>
    void operator() (Matrix& matrix)
    {
	print_matrix_row_cursor(matrix);
	cout << endl;
	Matrix sub_matrix(matrix.sub_matrix(2, 6, 2, 5));
	print_matrix_row_cursor(sub_matrix);

	typename traits::row<Matrix>::type                                 row(matrix);
	cout << row(*begin<glas::tags::nz_t>(sub_matrix)) << endl;

	cout << endl;
	print_matrix_row_cursor(matrix.sub_matrix(3, 5, 2, 4));
 
	recursion::matrix_recurator<Matrix> recurator(matrix);
	cout << endl;
	print_matrix_row_cursor(recurator.north_west().get_value());

	

	transposed_view<Matrix> trans_matrix(matrix);
	print_matrix_row_cursor(trans_matrix); 

    }
};

template <typename Matrix>
void fill_matrix(Matrix& matrix)
{
    typename traits::row<Matrix>::type                                 row(matrix);
    typename traits::col<Matrix>::type                                 col(matrix);
    typename traits::value<Matrix>::type                               value(matrix);
    typedef  glas::tags::nz_t                                          tag;
    typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;
    
    double x= 10.3;
    for (cursor_type cursor = begin<tag>(matrix), cend = end<tag>(matrix); cursor != cend; ++cursor) {
	value(*cursor, x);
	x+= 1.0; 
    }
       
}

 
int test_main(int argc, char* argv[])
{
    typedef morton_dense<double,  0x55555555, matrix_parameters<> > matrix_type;    
    matrix_type matrix(non_fixed::dimensions(16, 15));
   
    fill_matrix(matrix);

    test_morton_dense()(matrix);
    return 0;
}
