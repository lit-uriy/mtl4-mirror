// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/compressed2D.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/maybe.hpp>
#include <boost/numeric/mtl/operations/raw_copy.hpp>
#include <boost/numeric/mtl/operations/update.hpp>

using namespace mtl;
using namespace std;

template <typename Matrix>
void print_matrix(Matrix const& matrix)
{
    for (size_t r = 0; r < matrix.num_rows(); ++r) {
	cout << '[';
	for (size_t c = 0; c < matrix.num_cols(); ++c) {
	    cout << matrix(r, c) 
		 << (c < matrix.num_cols() - 1 ? ", " : "]\n"); } 
    }    
}

int test_main(int argc, char* argv[])
{
    typedef matrix_parameters<row_major, mtl::index::c_index, non_fixed::dimensions>   parameters;
    typedef compressed2D<int, parameters>                                              matrix_type;
    matrix_type   matrix(non_fixed::dimensions(8, 6)); 
	
    size_t        sts[] = {0, 2, 3, 7, 9, 12, 16, 16, 18},
                  ind[] = {1, 4, 2, 0, 1, 2, 3, 3, 5, 1, 3, 4, 1, 3, 4, 5, 2, 5};
    int           val[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    matrix.raw_copy(val, val+18, sts, ind);

    print_matrix(matrix); 
    if (matrix(1, 3) != 0) throw "Error in raw_copy, should be empty";
    if (matrix(2, 3) != 7) throw "Error in raw_copy, should be 7";


    {   // Inserter that overwrites the old values
	compressed2D_inserter<int, parameters>  i1(matrix, 3);

	i1(0, 3) << 31; i1(3, 3) << 33; i1(6, 0) << 34 << 35; i1(4, 4) << 36 << 37;
    }

    cout << "\n\n";
    print_matrix(matrix); 
    if (matrix(0, 3) != 31) throw "Error overwriting empty value";
    if (matrix(3, 3) != 33) throw "Error overwriting existing value";
    if (matrix(6, 0) != 35) throw "Error overwriting empty value twice";
    if (matrix(4, 4) != 37) throw "Error overwriting existing value twice";

    {   // Inserter that adds to the old values
        compressed2D_inserter<int, parameters, operations::update_add<int> > i2(matrix, 3);    
 
	i2(2, 2) << 21; i2(2, 4) << 22; i2(6, 1) << 23; 
	i2(7, 2) << 24 << 2; i2(4, 2) << 25; i2(2, 5) << 26; 
	i2(0, 2) << 27; i2(3, 1) << 28; i2(4, 2) << 29; 
    }

    cout << "\n\n";
    print_matrix(matrix); 
    if (matrix(0, 2) != 27) throw "Error adding to empty value";
    if (matrix(2, 2) != 27) throw "Error adding to existing value";
    if (matrix(4, 2) != 54) throw "Error adding to existing value twice (in 2 statements)";
    if (matrix(7, 2) != 43) throw "Error adding to existing value twice (in 1 statement)";

    return 0;
}
 
