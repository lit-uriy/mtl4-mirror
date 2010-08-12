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
#include <vector>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;


template <typename Vector>
void test(const Vector& v, const char* name)
{
    mtl::multi_vector<Vector> A(5, 5), B(2, 3);
    
    A[2][3]= 7.0;
    cout << "A[2][3] = " << A[2][3] << "\n";

    A= 3.0;
    cout << "A after initialization is \n" << A;

    if (A[1][1] != 3.0) throw "Wrong value in diagonal";
    if (A[1][2] != 0.0) throw "Wrong value off diagonal";

    cout << "Dimension of B is " << num_rows(B) << " x " << num_cols(B) << "\n\n";
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;

    dense_vector<double>    v;
    std::vector<double>     w;

    test(v, "dense_vector<double>");
    test(w, "std::vector<double>");

    return 0;
}
