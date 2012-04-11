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
#include <utility>
#include <cmath>
// #include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <boost/numeric/itl/smoother/gauss_seidel.hpp>

using namespace std;  
   
int main(int, char**)
{
    using namespace mtl;

    typedef mtl::dense_vector<double> Vector;
    typedef mtl::dense2D<double> Matrix;
    Vector       x(9, 8), b(9);
    Matrix   A(9,9);
    laplacian_setup(A, 3, 3);
    std::cout<< "x= " << x << "\n";
    
    b= A*x;
    x= 0;

    itl::gauss_seidel<Matrix, Vector > gs(A,b);
    for (int i =0 ; i< 30; i++){
            x=gs(x);
    }
    
    std::cout<< "x=" << x << "\n";
    Vector tmp(b-A*x);
    assert(two_norm(tmp) < 1.0e-4);
    
    
    return 0;
}
 














