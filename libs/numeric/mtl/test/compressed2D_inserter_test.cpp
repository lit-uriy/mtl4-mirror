// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

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


using namespace std;



template <typename Orientation, typename Indexing>
void test_compressed2D_insertion()
{
    typedef mtl::matrix::parameters<Orientation, Indexing, mtl::fixed::dimensions<8, 6> >         parameters;
    typedef mtl::compressed2D<int, parameters>                                              matrix_type;
    matrix_type   matrix; 

    {
	mtl::matrix::compressed2D_inserter<int, parameters>  i0(matrix, 3);
	i0(2, 2) << 6; i0(7, 2) << 17; 
    }
    {   // Inserter that overwrites the old values
	mtl::matrix::compressed2D_inserter<int, parameters>  i1(matrix, 3);

	i1(0, 3) << 31; i1(3, 3) << 33; i1(6, 0) << 34 << 35; i1(4, 4) << 36 << 37;
    }

    cout << "\n\n";
    print_matrix(matrix); 
    if (matrix(0, 3) != 31) throw "Error overwriting empty value";
    if (matrix(3, 3) != 33) throw "Error overwriting existing value";
    if (matrix(6, 0) != 35) throw "Error overwriting empty value twice";
    if (matrix(4, 4) != 37) throw "Error overwriting existing value twice";

    {   // Inserter that adds to the old values
        mtl::matrix::compressed2D_inserter<int, parameters, mtl::operations::update_plus<int> > i2(matrix, 3);    
 
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
    cout << "\n\n";
    {
	mtl::matrix::inserter<matrix_type, mtl::operations::update_plus<int> >  i3(matrix, 7);
	i3(2, 2) << 1;
    }

    if (matrix(2, 2) != 28) throw "Error adding to existing value";
}
 
int test_main(int argc, char* argv[])
{
    test_compressed2D_insertion<mtl::row_major, mtl::index::c_index>();
    test_compressed2D_insertion<mtl::col_major, mtl::index::c_index>();

    return 0;
}
