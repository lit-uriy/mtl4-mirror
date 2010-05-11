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
#include <boost/utility.hpp>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/operation/svd.hpp>

using namespace std;
int test_main(int argc, char* argv[])
{
    using namespace mtl;
    unsigned size=12, row= size, col=size;

    double b, normA(0), tol(0.0000001);
    compressed2D<double>                    A(row, col), A_t(row, col), S(row, row), V(row, col), D(col,col), norm(row, col),
					    AT(col,row), A_tT(col,row), ST(col, col), VT(col,row), DT(row,row), normT(col, row);
    A= 0;
    laplacian_setup(A,3,4);

    std::cout<<"A=\n"<< A <<"\n";
    AT= trans(A);
    std::cout<<"START--------------\n";

    boost::tie(S, V, D)= svd(A, tol);
    std::cout<<"MAtrix  S=\n"<< S <<"\n";
    std::cout<<"MAtrix  V=\n"<< V <<"\n";
    std::cout<<"MAtrix  D=\n"<< D <<"\n";
    A_t= S*V*trans(D);
    std::cout<<"MAtrix  A=S*V*D'=\n"<< A_t <<"\n";
//     std::cout<<"Original A==\n"<< A <<"\n";
    norm= A_t - A;
    normA= one_norm(norm);
    std::cout<< "norm(SVD-A)=" << normA << "\n";
     if (normA > size*size*tol) throw mtl::logic_error("wrong SVD decomposition of matrix A");
    return 0;
}

