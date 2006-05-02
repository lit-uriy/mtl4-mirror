// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/compressed2D.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/mtl_exception.hpp>
#include <boost/numeric/mtl/utilities/maybe.hpp>
#include <boost/numeric/mtl/operations/print_matrix.hpp>

using namespace mtl;
using namespace std;


template <typename Matrix>
void one_d_iteration(char const* name, Matrix & matrix)
{
    typename traits::row<Matrix>::type                                 row(matrix);
    typename traits::col<Matrix>::type                                 col(matrix);
    typename traits::value<Matrix>::type                               value(matrix);
    typedef  glas::tags::nz_t                                          tag;
    typedef typename traits::range_generator<tag, Matrix>::type        cursor_type;
    typedef typename traits::range_generator<tag, Matrix>::complexity  complexity;
    
    cout << name << "\nElements: " << complexity() << '\n';
    for (cursor_type cursor(begin<tag>(matrix)), cend(end<tag>(matrix)); cursor != cend; ++cursor) {
	cout << "matrix[" << row(*cursor) << ", " << col(*cursor) << "] = " << value(*cursor) << '\n';
	if (row(*cursor) == 2 && col(*cursor) == 2 && value(*cursor) != 7)
	    throw test_exception();
	if (row(*cursor) == 2 && col(*cursor) == 4 && value(*cursor) != 0)
	    throw test_exception();
    }
}

template <typename Matrix>
void matrix_init(Matrix& matrix)
{
    typedef typename Matrix::parameters   parameters;
    typedef typename Matrix::value_type   value_type;

    compressed2D_inserter<value_type, parameters> inserter(matrix);
    inserter(2, 2) << 7; inserter(1, 4) << 3; inserter(3, 2) << 9; inserter(5, 1) << 5;
}
    
 
template <typename Orientation, typename Indexing>
void test_compressed2D()
{
    typedef matrix_parameters<Orientation, Indexing, fixed::dimensions<8, 6> >         parameters;
    typedef compressed2D<int, parameters>                                              matrix_type;
    matrix_type                                                                        matrix; 

    matrix_init(matrix);
    std::cout << "\n\n";
    print_matrix(matrix);

    one_d_iteration("\nMatrix", matrix);


    transposed_view<matrix_type> trans_matrix(matrix);
    print_matrix(trans_matrix);

    one_d_iteration("\nTransposed matrix", trans_matrix);

};

int test_main(int argc, char* argv[])
{
    test_compressed2D<row_major, mtl::index::c_index>();
    test_compressed2D<row_major, mtl::index::f_index>();
    test_compressed2D<col_major, mtl::index::c_index>();
    test_compressed2D<col_major, mtl::index::f_index>();
    
    return 0;
}
 
