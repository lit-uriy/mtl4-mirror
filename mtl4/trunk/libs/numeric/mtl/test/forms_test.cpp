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
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


template <typename ResMatrix, typename ArgMatrix>
void test(const ResMatrix&, const ArgMatrix& B)
{
    ResMatrix C(B * B);

    C+= trans(B) * B;
    C+= trans(B) * B * B;
    C+= trans(B) * 3.5 * B * B;
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    typedef vector::parameters<tag::col_major, vector::fixed::dimension<2>, true> fvec_para;
    typedef matrix::parameters<tag::row_major, mtl::index::c_index, mtl::fixed::dimensions<2, 2>, true> fmat_para;

    float ma[2][2]= {{2., 3.}, {4., 5.}}, va[2]= {3., 4.};
    
    dense2D<float>                   A_dyn(ma);
    dense2D<float, fmat_para>        A_stat(ma);
    dense_vector<float>              v_dyn(va);
    dense_vector<float, fvec_para>   v_stat(va);

    test(A_dyn, A_dyn);
    test(A_dyn, A_stat);

    return 0;
}

