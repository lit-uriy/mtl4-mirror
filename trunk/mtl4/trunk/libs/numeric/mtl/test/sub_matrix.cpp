// $COPYRIGHT$

#include <iostream>
#include <string>
#include <boost/test/minimal.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp>
#include <boost/numeric/mtl/matrix/transposed_view.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/operation/sub_matrix.hpp>
#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/mtl/recursion/for_each.hpp>


using namespace mtl;
using namespace std;  


template <typename Recurator>
void print_depth_first(Recurator const& recurator, string str)
{
    cout << "\nRecursion: " << str << endl;
    print_matrix_row_cursor(recurator.get_value());
  
    // for full recursion remove the string length limitation
    if (!recurator.is_leaf()) { // && str.length() < 20) {     
	if (!recurator.north_west_empty())
	    print_depth_first(recurator.north_west(), string("north west of ") + str);
	if (!recurator.south_west_empty())
	    print_depth_first(recurator.south_west(), string("south west of ") + str);
	if (!recurator.north_east_empty())
	    print_depth_first(recurator.north_east(), string("north east of ") + str);
	if (!recurator.south_east_empty())
	    print_depth_first(recurator.south_east(), string("south east of ") + str);
    }
} 


template <typename Recurator, typename BaseCaseTest>
void recursive_print(Recurator const& recurator, string str, BaseCaseTest const& is_base)
{
    if (is_base(recurator)) {
	cout << "\nBase case: " << str << endl;
	print_matrix_row_cursor(recurator.get_value());
    } else {
	recursive_print(recurator.north_west(), string("north west of ") + str, is_base);
	recursive_print(recurator.south_west(), string("south west of ") + str, is_base);
	recursive_print(recurator.north_east(), string("north east of ") + str, is_base);
	recursive_print(recurator.south_east(), string("south east of ") + str, is_base);
    }
} 


template <typename Recurator, typename BaseCaseTest>
void recursive_print_checked(Recurator const& recurator, string str, BaseCaseTest const& is_base)
{
    if (is_base(recurator)) {
	cout << "\nBase case: " << str << endl;
	print_matrix_row_cursor(recurator.get_value());
    } else {
	if (!recurator.north_west_empty())
	    recursive_print_checked(recurator.north_west(), string("north west of ") + str, is_base);
	if (!recurator.south_west_empty())
	    recursive_print_checked(recurator.south_west(), string("south west of ") + str, is_base);
	if (!recurator.north_east_empty())
	    recursive_print_checked(recurator.north_east(), string("north east of ") + str, is_base);
	if (!recurator.south_east_empty())
	    recursive_print_checked(recurator.south_east(), string("south east of ") + str, is_base);
    }
} 

struct print_functor
{
    template <typename Matrix>
    void operator() (Matrix const& matrix) const
    {
	print_matrix_row_cursor(matrix);
	cout << endl;
    }
};

template <typename Matrix>
void test_sub_matrix(Matrix& matrix)
{
    using recursion::for_each;

    print_matrix_row_cursor(matrix);
    
    // recursion::min_dim_test             is_base(2);
    // recursion::undivisible_min_dim_test is_base(2);
    recursion::max_dim_test             is_base(2);
    recursion::matrix_recurator<Matrix> recurator(matrix);
    // print_depth_first(recurator, "");
    recursive_print_checked(recurator, "", is_base);
	 
#if 0 
    cout << "\n====================\n"
	 <<   "Same with transposed\n"
	 <<   "====================\n\n";

    transposed_view<Matrix> trans_matrix(matrix);

    print_matrix_row_cursor(trans_matrix); 
    recursion::matrix_recurator< transposed_view<Matrix> > trans_recurator(trans_matrix);
    // print_depth_first(trans_recurator, "");
    recursive_print_checked(trans_recurator, "", is_base);
	 
    cout << "\n=============================\n"
	 <<   "Again with recursive for_each\n"
	 <<   "=============================\n\n";

    recursion::for_each(trans_recurator, print_functor(), is_base);
#endif
}


template <typename Matrix>
void fill_matrix(Matrix& matrix)
{
    typename traits::row<Matrix>::type                                 row(matrix);
    typename traits::col<Matrix>::type                                 col(matrix);
    typename traits::value<Matrix>::type                               value(matrix);
    typedef  glas::tag::nz                                          tag;
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

    typedef morton_dense<double,  0x55555555> matrix_type;    
    matrix_type matrix(6, 5);   
    fill_matrix(matrix); 
    test_sub_matrix(matrix);

    cout << "\n=========================\n"
	 << "Doppler matrix (4x4 base)\n"
	 << "=========================\n\n";

    typedef morton_dense<double,  0x55555553> dmatrix_type;    
    dmatrix_type dmatrix(6, 5);   
    fill_matrix(dmatrix); 
    test_sub_matrix(dmatrix);

    cout << "\n======================\n"
	 << "Row-major dense matrix\n"
	 << "======================\n\n";

    dense2D<double, matrix_parameters<> >   rmatrix(non_fixed::dimensions(6, 5));
    fill_matrix(rmatrix); 
    test_sub_matrix(rmatrix);
 
    cout << "=================================\n"
	 << "Vector-like morton-ordered matrix\n"
	 << "=================================\n\n";

    matrix_type vmatrix(17, 2);   
    fill_matrix(vmatrix); 
    test_sub_matrix(vmatrix);

    return 0;
}

