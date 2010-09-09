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
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/utility/is_static.hpp>


int test_main(int argc, char* argv[])
{
    using namespace mtl;

    typedef matrix::parameters<tag::row_major, mtl::index::c_index, mtl::fixed::dimensions<3, 3>, true> mat_para;
    typedef vector::parameters<tag::col_major, vector::fixed::dimension<3>, true>                       vec_para;

    if ( traits::is_static<mtl::non_fixed::dimensions>::value) throw "Must not be static!";
    if (!traits::is_static<mtl::fixed::dimensions<1, 2> >::value) throw "Must be static!";

    if ( traits::is_static<mtl::vector::non_fixed::dimension>::value) throw "Must not be static!";
    if (!traits::is_static<mtl::vector::fixed::dimension<1> >::value) throw "Must be static!";

    if ( traits::is_static<mtl::dense2D<float> >::value) throw "Must not be static!";
    if (!traits::is_static<mtl::dense2D<float, mat_para> >::value) throw "Must be static!";

    if ( traits::is_static<mtl::morton_dense<float, morton_mask> >::value) throw "Must not be static!";
    if (!traits::is_static<mtl::morton_dense<float, morton_mask, mat_para> >::value) throw "Must be static!";

    if ( traits::is_static<mtl::compressed2D<float> >::value) throw "Must not be static!";
    if (!traits::is_static<mtl::compressed2D<float, mat_para> >::value) throw "Must be static!";

    if ( traits::is_static<mtl::dense_vector<float> >::value) throw "Must not be static!";
    if (!traits::is_static<mtl::dense_vector<float, vec_para> >::value) throw "Must be static!";

    return 0;
}
