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

// With contributions from Cornelius Steinhardt

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/operation/cuppen.hpp>

using namespace std;


int test_main(int , char**)
{
    using namespace mtl;

    dense_vector<double>                    eig;

    double array[][4]= {{1,  2,   0,  0},
                        {2, -1,  -2,  0},
                        {0, -2,   1,  3},
                        {0,  0,   3, 10}};
    dense2D<double> A(array), Q(4,4), L(4,4);
    std::cout << "A=\n" << A << "\n";

    eig= eigenvalue_symmetric(A,22);

    std::cout<<"eigenvalues  ="<< eig <<"\n";
    
    cuppen(A, Q, L);
    std::cout<<"A  =\n"<< A <<"\n";
    std::cout<<"Q  =\n"<< Q <<"\n";
    std::cout<<"L  =\n"<< L <<"\n";
    
    

    return 0;
}



