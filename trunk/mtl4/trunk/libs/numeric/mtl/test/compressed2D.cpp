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
#include <boost/numeric/mtl/complexity.hpp>
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

#if 0
template <typename Matrix>
void one_d_iterator_iteration(char const* name, Matrix & matrix)
{
    typedef  glas::tags::nz_cit                                          tag;
    typedef typename traits::range_generator<tag, Matrix>::type        iterator_type;
    typedef typename traits::range_generator<tag, Matrix>::complexity  complexity;
    
    cout << name << "\nElements: " << complexity() << '\n';
    for (iterator_type iterator(begin<tag>(matrix)), cend(end<tag>(matrix)); iterator != cend; ++iterator) 
	cout << *iterator << '\n';
}
#endif


template <typename Matrix, typename Tag, typename Complexity>
void two_d_iteration_impl(char const* outer, Matrix & matrix, Tag, Complexity)
{
    typename traits::row<Matrix>::type                                 row(matrix); 
    typename traits::col<Matrix>::type                                 col(matrix); 
    typename traits::const_value<Matrix>::type                         value(matrix); 
    typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;
    // typedef typename traits::range_generator<Tag, Matrix>::complexity  complexity;

    cout << outer << ": " << Complexity() << '\n';
    // check_same_type(complexity(), ExpComplexity());
    for (cursor_type cursor = begin<Tag>(matrix), cend = end<Tag>(matrix); cursor != cend; ++cursor) {
	typedef glas::tags::nz_t     inner_tag;
	cout << "---\n";
	typedef typename traits::range_generator<inner_tag, cursor_type>::type icursor_type;
	for (icursor_type icursor = begin<inner_tag>(cursor), icend = end<inner_tag>(cursor); icursor != icend; ++icursor)
	    cout << "matrix[" << row(*icursor) << ", " << col(*icursor) << "] = " << value(*icursor) << '\n';
    }
} 


template <typename Matrix, typename Tag>
void two_d_iteration_impl(char const* name, Matrix & matrix, Tag, complexity_classes::infinite)
{
    cout << name << ": Tag has no implementation\n";
}

template <typename Matrix, typename Tag, typename Complexity>
void two_d_iterator_iteration_impl(char const* outer, Matrix & matrix, Tag, Complexity)
{
    typename traits::row<Matrix>::type                                 row(matrix); 
    typename traits::col<Matrix>::type                                 col(matrix); 
    typename traits::const_value<Matrix>::type                         value(matrix); 
    typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;

    cout << outer << " with iterators: " << Complexity() << '\n';
    for (cursor_type cursor = begin<Tag>(matrix), cend = end<Tag>(matrix); cursor != cend; ++cursor) {
	typedef glas::tags::nz_cit     inner_tag;
	cout << "---\n";
	typedef typename traits::range_generator<inner_tag, cursor_type>::type iter_type;
	for (iter_type iter = begin<inner_tag>(cursor), i_end = end<inner_tag>(cursor); iter != i_end; ++iter)
	    cout << *iter << '\n';
    }
} 


template <typename Matrix, typename Tag>
void two_d_iterator_iteration_impl(char const* name, Matrix & matrix, Tag, complexity_classes::infinite)
{
    cout << name << ": Tag has no implementation\n";
}

template <typename Matrix, typename Tag>
void two_d_iteration(char const* name, Matrix & matrix, Tag)
{
    typedef typename traits::range_generator<Tag, Matrix>::complexity  complexity;
    two_d_iteration_impl(name, matrix, Tag(), complexity());
    two_d_iterator_iteration_impl(name, matrix, Tag(), complexity());
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
void test_compressed2D(char const* name)
{
    cout << "\n====================\n" << name << "\n====================\n";
    typedef matrix_parameters<Orientation, Indexing, fixed::dimensions<8, 6> >         parameters;
    typedef compressed2D<int, parameters>                                              matrix_type;
    matrix_type                                                                        matrix; 

    matrix_init(matrix);
    std::cout << "\n\n";
    print_matrix(matrix);

    one_d_iteration("\nMatrix", matrix); 
    //one_d_iterator_iteration("\nMatrix (iterator)", matrix); 

    two_d_iteration("Row-wise", matrix, glas::tags::row_t());
    two_d_iteration("Column-wise", matrix, glas::tags::col_t());
    two_d_iteration("On Major", matrix, glas::tags::major_t());


    transposed_view<matrix_type> trans_matrix(matrix);
    cout << "\n===\n";
    print_matrix(trans_matrix);

    one_d_iteration("\nTransposed matrix", trans_matrix);
    //one_d_iterator_iteration("\nMatrix (iterator)", trans_matrix); 

    two_d_iteration("Transposed row-wise", trans_matrix, glas::tags::row_t());
    two_d_iteration("Transposed Column-wise", trans_matrix, glas::tags::col_t());
    two_d_iteration("Transposed On Major", trans_matrix, glas::tags::major_t());

};

int test_main(int argc, char* argv[])
{
    test_compressed2D<row_major, mtl::index::c_index>("CRS C");
    test_compressed2D<row_major, mtl::index::f_index>("CRS F");
    test_compressed2D<col_major, mtl::index::c_index>("CCS C");
    test_compressed2D<col_major, mtl::index::f_index>("CCS F");
    
    return 0;
}
 
