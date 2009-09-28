// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int test_main(int argc, char* argv[])
{
    using namespace mtl;

    typedef mtl::fixed::dimensions<3, 3> fmdim;
    typedef mtl::non_fixed::dimensions   mdim;

    typedef mtl::vector::fixed::dimension<3>    fvdim;
    typedef mtl::vector::non_fixed::dimension   vdim;

    typedef matrix::parameters<tag::row_major, mtl::index::c_index, mdim>   mat_para;
    typedef matrix::parameters<tag::row_major, mtl::index::c_index, fmdim>  fmat_para;

    typedef vector::parameters<tag::col_major, vdim>                        vec_para;
    typedef vector::parameters<tag::col_major, fvdim>                       fvec_para;

    if ( mat_para::on_stack) throw "Must not be on stack!";
    if (!fmat_para::on_stack) throw "Must be on stack!";

    if ( vec_para::on_stack) throw "Must not be on stack!";
    if (!fvec_para::on_stack) throw "Must be on stack!";

    return 0;
}
