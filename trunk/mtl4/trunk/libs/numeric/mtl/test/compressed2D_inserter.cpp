// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/type_traits.hpp>

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/transposed_view.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/glas_tag.hpp>
#include <boost/numeric/mtl/detail/index.hpp>
#include <boost/numeric/mtl/utility/maybe.hpp>
#include <boost/numeric/mtl/operation/raw_copy.hpp>
#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/mtl/operation/print_matrix.hpp>

using namespace mtl;
using namespace std;

template <typename Matrix>
void raw_copy_test(Matrix& matrix, row_major)
{
    size_t        sts[] = {0, 2, 3, 7, 9, 12, 16, 16, 18},
                  ind[] = {1, 4, 2, 0, 1, 2, 3, 3, 5, 1, 3, 4, 1, 3, 4, 5, 2, 5};
    int           val[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    matrix.raw_copy(val, val+18, sts, ind);
}


template <typename Matrix>
void raw_copy_test(Matrix& matrix, col_major)
{
    size_t        sts[] = {0, 1, 5, 8, 12, 15, 18},
                  ind[] = {2, 0, 2, 4, 5, 1, 2, 7, 2, 3, 44, 5, 0, 4, 5, 5, 7};
    int           val[] = {4, 1, 5, 10, 13, 3, 6, 17, 7, 8, 11, 14, 2, 12, 15, 16, 18};
    matrix.raw_copy(val, val+18, sts, ind);
}


template <typename Orientation, typename Indexing>
void test_compressed2D_insertion()
{
    typedef matrix::parameters<Orientation, Indexing, fixed::dimensions<8, 6> >         parameters;
    typedef compressed2D<int, parameters>                                              matrix_type;
    matrix_type   matrix; 
    const int io= mtl::index::change_to(Indexing(), 0);  // index offset 1 for Fortran, 0 for C

    raw_copy_test(matrix, Orientation());
    print_matrix(matrix); 
    if (matrix(1+io, 3+io) != 0) throw "Error in raw_copy, should be empty";
    if (matrix(2+io, 3+io) != 7) throw "Error in raw_copy, should be 7";

    {   // Inserter that overwrites the old values
	compressed2D_inserter<int, parameters>  i1(matrix, 3);

	i1(0+io, 3+io) << 31; i1(3+io, 3+io) << 33; i1(6+io, 0+io) << 34 << 35; i1(4+io, 4+io) << 36 << 37;
    }

    cout << "\n\n";
    print_matrix(matrix); 
    if (matrix(0+io, 3+io) != 31) throw "Error overwriting empty value";
    if (matrix(3+io, 3+io) != 33) throw "Error overwriting existing value";
    if (matrix(6+io, 0+io) != 35) throw "Error overwriting empty value twice";
    if (matrix(4+io, 4+io) != 37) throw "Error overwriting existing value twice";

    {   // Inserter that adds to the old values
        compressed2D_inserter<int, parameters, operations::update_plus<int> > i2(matrix, 3);    
 
	i2(2+io, 2+io) << 21; i2(2+io, 4+io) << 22; i2(6+io, 1+io) << 23; 
	i2(7+io, 2+io) << 24 << 2; i2(4+io, 2+io) << 25; i2(2+io, 5+io) << 26; 
	i2(0+io, 2+io) << 27; i2(3+io, 1+io) << 28; i2(4+io, 2+io) << 29; 
    }

    cout << "\n\n";
    print_matrix(matrix); 
    if (matrix(0+io, 2+io) != 27) throw "Error adding to empty value";
    if (matrix(2+io, 2+io) != 27) throw "Error adding to existing value";
    if (matrix(4+io, 2+io) != 54) throw "Error adding to existing value twice (in 2 statements)";
    if (matrix(7+io, 2+io) != 43) throw "Error adding to existing value twice (in 1 statement)";
    cout << "\n\n";
 
    {
	matrix::inserter<matrix_type, operations::update_plus<int> >  i3(matrix, 7);
	i3(2+io, 2+io) << 1;
    }
    if (matrix(2+io, 2+io) != 28) throw "Error adding to existing value";
    cout << "\nmatrix[2][2] = " << matrix[2+io][2+io] << "\n";
}
 
int test_main(int argc, char* argv[])
{
    test_compressed2D_insertion<row_major, mtl::index::c_index>();
    test_compressed2D_insertion<row_major, mtl::index::f_index>();
    test_compressed2D_insertion<col_major, mtl::index::c_index>();
    test_compressed2D_insertion<col_major, mtl::index::f_index>();

    return 0;
}
