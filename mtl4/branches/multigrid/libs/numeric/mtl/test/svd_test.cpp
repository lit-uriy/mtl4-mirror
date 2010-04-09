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
    unsigned size=3, row= size, col=size;

    double b;
    dense_vector<double>                    vec(size), vec1(size);
    dense2D<double>                                      dr(row, col), dr_t(row, col), S(row, row), V(row, col), D(col,col);
    dr= 0;

    dr[0][0]=1;
    dr[0][1]=1;
    dr[0][2]=1;
    //dr[0][3]=4;
    dr[1][0]=1;
    dr[1][1]=2;
    dr[1][2]=2;
    //dr[1][3]=3;
    dr[2][0]=9;
    dr[2][1]=3;
    dr[2][2]=2;
    //dr[2][3]=4;
    std::cout<<"A=\n"<< dr <<"\n";
    std::cout<<"START--------------\n";

  
    boost::tie(S, V, D)= svd(dr, 0.0000001);
    std::cout<<"MAtrix  S=\n"<< S <<"\n";
    std::cout<<"MAtrix  V=\n"<< V <<"\n";
    std::cout<<"MAtrix  D=\n"<< D <<"\n";
    
//    std::cout<<"MAtrix  A=\n"<< dr <<"\n";
	dr_t= S*V*trans(D);
    std::cout<<"MAtrix  A=S*V*D'=\n"<< dr_t <<"\n";
    std::cout<<"Original A==\n"<< dr <<"\n";
    
    return 0;
}

