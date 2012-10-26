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
#include <vector>
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;

template <typename Vector>
void test(const char* name)
{
    mtl::multi_vector<Vector> A(4, 6), B(4, 6);
    A= 3.0;

    B= A;
    cout << "B= A yields\n" << B << endl;

    B= A + A;
    cout << "A + A is\n" << B << endl;

    B= 2.0 * A + 3 * A;
    cout << "2 * A + 3 * A is\n" << B << endl;

    B= 2.0 * A + 3 * A - 2.6 * A;
    cout << "2 * A + 3 * A - 2.6 * A is\n" << B << endl;
}

int main(int, char**)
{
    test<mtl::dense_vector<double> >("dense_vector<double>");

    return 0;
}
