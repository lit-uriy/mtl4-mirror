// $COPYRIGHT$

#include <iostream>
#include <string>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>
#include <boost/numeric/mtl/operations/sub_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>


using namespace mtl;
using namespace std;  
 

template <typename Recurator>
void print_depth_first(Recurator const& recurator, string str)
{
    cout << "\nRecursion: " << str << endl;
    print_matrix_row_cursor(recurator.get_value());
  
    // for full recursion remove the string length limitation
    if (!recurator.is_leaf() && str.length() < 20) {     
	print_depth_first(recurator.north_west(), string("north west of ") + str);
	print_depth_first(recurator.south_west(), string("south west of ") + str);
	print_depth_first(recurator.north_east(), string("north east of ") + str);
	print_depth_first(recurator.south_east(), string("south east of ") + str);
    }
} 

    template <typename Matrix>
    void test_sub_matrix(Matrix& matrix)
    {
	print_matrix_row_cursor(matrix);

	recursion::matrix_recurator<Matrix> recurator(matrix);
	print_depth_first(recurator, "");

	 
	cout << "\n====================\n"
	     <<   "Same with transposed\n"
	     <<   "====================\n\n";

	transposed_view<Matrix> trans_matrix(matrix);
	print_matrix_row_cursor(trans_matrix); 
	recursion::matrix_recurator< transposed_view<Matrix> > trans_recurator(trans_matrix);
	print_depth_first(trans_recurator, "");
    }


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

    cout << "=====================\n"
	 << "Morton-ordered matrix\n"
	 << "=====================\n\n";

    typedef morton_dense<double,  0x55555555, matrix_parameters<> > matrix_type;    
    matrix_type matrix(non_fixed::dimensions(6, 5));   
    fill_matrix(matrix); 
    test_sub_matrix(matrix);

    cout << "\n=====================\n"
	 << "Ordinary dense matrix\n"
	 << "=====================\n\n";

    dense2D<double, matrix_parameters<> >   dmatrix(non_fixed::dimensions(6, 5));
    fill_matrix(dmatrix); 
    test_sub_matrix(dmatrix);

    return 0;
}

 
